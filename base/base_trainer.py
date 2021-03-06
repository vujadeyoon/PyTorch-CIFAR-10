import os
import torch
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter
from model import model as module_arch
from vujade import vujade_slack as slack_
from vujade.vujade_debug import printf


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, criterions, metrics_ftn, optimizer, config, device):
        self.model = model
        self.criterions = criterions
        self.metrics_ftn = metrics_ftn
        self.optimizer = optimizer
        self.config = config
        self.device = device

        self.hpp = HyperParamsPerformance(_config=self.config)
        self.logger = self.config.get_logger('trainer', self.config['trainer']['verbosity'])
        cfg_trainer = self.config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')
        self.name_dataset = self.config['name']
        self.run_id = self.config.run_id
        self.slack = slack_.Slack(_token_usr=self.config['slack']['token_usr'],
                                  _token_bot=self.config['slack']['token_bot'],
                                  _channel=self.config['slack']['channel'],
                                  _is_time=self.config['slack']['is_time'],
                                  _is_debug=self.config['slack']['is_debug'])

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = self.config.save_dir

        # setup visualization writer instance
        self.writer = TensorboardWriter(self.config.log_dir, self.logger, self.config['visualization']['tensorboard'])
        self._add_graph()
        self._write_description()

        if self.config.resume is not None:
            self._resume_checkpoint(self.config.resume)

        self.num_epochs = (self.epochs + 1) - self.start_epoch

    def _add_graph(self):
        model = self.config.init_obj('arch', module_arch)
        tensor_size = self.config['train_loader']['args']['size'].copy()
        tensor_temp = torch.zeros(1, *tensor_size)
        self.writer.add_graph(model, tensor_temp)

    def _write_description(self):
        self.writer.add_text('Train', input('Input the description for {}: '.format(self.run_id)))

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as ckpt-best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                    self.hpp.update(_log=log)
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

        self.writer.add_hparams(self.hpp.get_hparams(), self.hpp.best, run_name=self.config['hparam']['run_name'])

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'ckpt-best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = os.path.join(self.checkpoint_dir, 'ckpt-epoch_{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'ckpt-best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: ckpt-best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = os.path.join(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))


class HyperParamsPerformance(object):
    def __init__(self, _config) -> None:
        super(HyperParamsPerformance, self).__init__()
        self.config = _config
        self.key_metric = self.config['hparam']['key_metric']
        self.best = {key: None for key in self.key_metric}

    def update(self, _log: dict):
        for _idx, (_key_metric) in enumerate(self.key_metric):
            self.best[_key_metric] = _log[_key_metric]

    def get_hparams(self) -> dict:
        try:
            res = {
                'name_dataset': self.config['name'],
                'model': self.config['arch']['type'],
                'epoch': self.config['trainer']['epochs'],
                'bsize_train': self.config['train_loader']['args']['batch_size'],
                'lrs_warmup_alogrithm': self.config['lr_scheduler']['warmup_alogrithm'],
                'lrs_warmup_epoch': self.config['lr_scheduler']['warmup_epoch'],
                'lrs_warmup_lr': self.config['lr_scheduler']['warmup_lr'],
                'lrs_name': self.config['lr_scheduler']['type'],
                'lrs_epoch': self.config['lr_scheduler']['args']['T_max'],
                'opt_name': self.config['optimizer']['type'],
                'opt_lr': self.config['optimizer']['args']['lr'],
                'opt_momentum': self.config['optimizer']['args']['momentum'],
                'opt_weight_decay': self.config['optimizer']['args']['weight_decay'],
                'loss_name': self.config['loss']['type'],
            }
        except Exception as e:
            raise ValueError('Some values of the _get_hparams() may be incorrect. Error: {}'.format(e))

        return res
