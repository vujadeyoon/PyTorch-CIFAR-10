import os
import cv2
import torch
import numpy as np
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from vujade import vujade_profiler as prof_
from vujade import vujade_path as path_
from vujade.vujade_debug import printf


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterions, metric_ftns, optimizer, config, device,
                 dataloader_train, dataloader_valid=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterions, metric_ftns, optimizer, config, device)
        self.dataloader_train = dataloader_train
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.dataloader_train)
        else:
            # iteration-based training
            self.dataloader_train = inf_loop(dataloader_train)
            self.len_epoch = len_epoch
        self.dataloader_valid = dataloader_valid
        self.do_validation = self.dataloader_valid is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = config['trainer']['log_step']
        self.num_iters = self.num_epochs * self.len_epoch
        self.time_eta = None
        self.eta = prof_.ETA(_len_epoch=self.len_epoch, _num_iters=self.num_iters, _warmup=0)
        self.num_iter_max = self.len_epoch * (self.epochs - 1) + self.len_epoch
        self.loss_wieght = np.cos(np.linspace(0, np.pi / 2, self.num_iter_max))

        self.metric_train = MetricTracker('loss', 'lr', *[m.__class__.__name__ for m in self.metric_ftns], writer=self.writer)
        self.metric_valid = MetricTracker('loss', 'lr', *[m.__class__.__name__ for m in self.metric_ftns], writer=self.writer)

        if self.config['trainer']['is_amp'] is True:
            self.scaler = torch.cuda.amp.GradScaler()

        self.params_vis = {
            'nrow': self.config['visualization']['nrow'],
            'ncol': self.config['visualization']['ncol'],
            'idx_picture': self.config['visualization']['idx_picture'],
            'dataformats': self.config['visualization']['dataformats'],
            'is_torchvision_norm': True if self.config['visualization']['normalize'] is True else False
        }
        if self.params_vis['is_torchvision_norm'] is False:
            self.params_vis['norm_mean'] = torch.as_tensor(self.config['visualization']['normalize']['mean'], dtype=torch.float32).view(-1, 1, 1)
            self.params_vis['norm_std'] = torch.as_tensor(self.config['visualization']['normalize']['std'], dtype=torch.float32).view(-1, 1, 1)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.metric_train.reset()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        for _idx_batch, (_data, _target) in enumerate(self.dataloader_train):
            self.eta.tic(_is_train=True)

            # Debug
            data, target = _data.to(self.device), _target.to(self.device)
            # printf('data: ', type(data), data.shape, data.dtype, data.device)           # [trainer.py: 69]: data:  <class 'torch.Tensor'> torch.Size([128, 3, 32, 32]) torch.float32 cuda:0
            # printf('target: ', type(target), target.shape, target.dtype, target.device) # [trainer.py: 70]: target:  <class 'torch.Tensor'> torch.Size([128]) torch.int64 cuda:0

            self.optimizer.zero_grad()

            num_iter_curr = self.len_epoch * (epoch - 1) + _idx_batch

            if self.config['trainer']['is_amp'] is True:
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    loss = self.criterions(output, target)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = self.criterions(output, target)
                loss.backward()
                self.optimizer.step()

            self.writer.set_step(num_iter_curr)
            self.metric_train.update('lr', self.optimizer.param_groups[0]['lr'])
            self.metric_train.update('loss', loss.item())
            for met in self.metric_ftns:
                self.metric_train.update(met.__class__.__name__, met(output, target))

            self.eta.toc(_is_train=True)
            self.time_eta = self.eta.get(_num_iter_curr=num_iter_curr)

            if _idx_batch % self.log_step == 0:
                self.logger.info('[{}: {}/{}] {} Loss: {:.6f}; ETA: {}.'.format(self.run_id, epoch, self.epochs, self._progress(_idx_batch), loss.item(), self.time_eta))

            if _idx_batch == self.len_epoch:
                break

        log = self.metric_train.result()

        if self.do_validation:
            self.eta.tic(_is_train=False)
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})
            self.eta.toc(_is_train=False)

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.metric_valid.reset()
        loss_cumsum = 0.0

        with torch.no_grad():
            for _idx_batch, (_data, _target) in enumerate(self.dataloader_valid):
                data, target = _data.to(self.device), _target.to(self.device)
                output = self.model(data)
                num_iter_curr = self.len_epoch * (epoch - 1) + self.len_epoch
                loss = self.criterions(output, target)
                loss_cumsum += loss.item()

                if _idx_batch < (self.params_vis['ncol'] * self.params_vis['nrow']):
                    if _idx_batch == 0:
                        _, _, tensor_vis_height, tensor_vis_width = data.shape
                        tensor_vis = torch.zeros(size=(self.params_vis['nrow'] * self.params_vis['ncol'], 3, tensor_vis_height, tensor_vis_width), dtype=torch.uint8)

                    tensor_vis[_idx_batch, :, :, :] = torch.clamp(255.0 * _data[self.params_vis['idx_picture']].mul_(self.params_vis['norm_std']).add_(self.params_vis['norm_mean']), 0.0, 255.0).type(torch.uint8)

                self.writer.set_step((epoch - 1) * len(self.dataloader_valid) + _idx_batch, 'valid')

        # Update metric
        self.writer.set_step(epoch, 'valid')
        self.metric_valid.update('lr', self.optimizer.param_groups[0]['lr'])
        self.metric_valid.update('loss', loss_cumsum / len(self.dataloader_valid))
        for met in self.metric_ftns:
            self.metric_valid.update(met.__class__.__name__, met(output, target))

        self.writer.add_image('data', make_grid(tensor_vis, nrow=self.params_vis['ncol'], normalize=self.params_vis['is_torchvision_norm']), dataformats=self.params_vis['dataformats'])

        # Post messages for the slack
        if self.config['slack']['is_slack'] is True:
            msg = '[Valid: {}/{}] Loss: {:.2f}; '.format(epoch, self.epochs, loss_cumsum / len(self.dataloader_valid))
            for met in self.metric_ftns:
                msg += '{}: {}; '.format(met.__class__.__name__, met(output, target))
            msg += 'ETA: {}.'.format(self.time_eta)

        # Add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        return self.metric_valid.result()

    def _progress(self, _idx_batch):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.dataloader_train, 'n_samples'):
            current = _idx_batch * self.dataloader_train.batch_size
            total = self.dataloader_train.n_samples
        else:
            current = _idx_batch
            total = self.len_epoch
        return base.format(current +  1, total, 100.0 * (current + 1) / total)