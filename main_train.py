import os
import argparse
import collections
import torch
import dataloader.classification as module_dataloader
import loss.classification as module_loss
import metric.classification as module_metric
from model import model as module_arch
from config.parse_config import ConfigParser
from trainer import LRScheduler, Trainer
from utils import prepare_device
from vujade import vujade_dnn as dnn_
from vujade import vujade_path as path_
from vujade import vujade_utils as utils_
from vujade.vujade_debug import printf


def backup(_cmd: str, _spath_backup: str) -> bool:
    is_success = utils_.SystemCommand.run(_command=_cmd, _is_daemon=False, _is_subprocess=True)
    path_backup = path_.Path(_spath=_spath_backup)
    if (is_success & path_backup.path.is_file()) is True:
        res = True
    else:
        res = False

    return res


def _main(config):
    logger = config.get_logger('train')

    if config['backup']['is_backup'] is True:
        spath_backup = os.path.join(config.save_dir, 'backup_{}.tar.gz'.format(config.run_id))
        cmd = config['backup']['cmd'].replace('{SPATH_BACKUP}', spath_backup)
        if backup(_cmd=cmd, _spath_backup=spath_backup):
            logger.info('The backup is successful to {}.'.format(spath_backup))
        else:
            logger.info('The backup failed to {}.'.format(spath_backup))

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(is_cuda=config['trainer']['is_cuda'], n_gpu_use=config['n_gpu'])

    # fix random seeds for reproducibility
    utils_.set_seed(_device=device, _seed=config['trainer']['seed'])

    # setup data_loader instances
    dataloader_train = config.init_obj('train_loader', module_dataloader).loader
    dataloader_valid = config.init_obj('valid_loader', module_dataloader).loader

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)

    if config['summary']['model']['is_print']:
        logger.info(model)

    if config['summary']['computational_cost']['is_print']:
        macs, flops, params = dnn_.CalculateComputationalCost(
            _units_macs=config['summary']['computational_cost']['units_macs'],
            _units_flops=config['summary']['computational_cost']['units_flops'],
            _units_params=config['summary']['computational_cost']['units_params'],
            _precision=config['summary']['computational_cost']['precision']).run(
            _model=model,
            _input_res=tuple(config['summary']['computational_cost']['input_resolution']),
            _print_per_layer_stat=config['summary']['computational_cost']['print_per_layer_stat'],
            _as_strings=config['summary']['computational_cost']['as_strings'],
            _verbose=config['summary']['computational_cost']['verbose'])

        logger.info('{:<25} {}.'.format('Tensor shape: ', tuple(config['summary']['computational_cost']['input_resolution'])))
        logger.info('{:<25} {}.'.format('Trainable parameters: ', params))
        logger.info('{:<25} {}.'.format('Macs: ', macs))
        logger.info('{:<25} {}.'.format('Flops: ', flops))

    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterions = config.init_obj('loss', module_loss).to(device)
    metrics_ftn = [config.init_obj(('metric', met), module_metric) for met in config['metric']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = LRScheduler(_config=config,
                               _optimizer=optimizer,
                               _warmup_alogrithm=config['lr_scheduler']['warmup_alogrithm'],
                               _warmup_epoch=config['lr_scheduler']['warmup_epoch'],
                               _warmup_lr=config['lr_scheduler']['warmup_lr'])

    trainer = Trainer(model=model,
                      criterions=criterions,
                      metric_ftns=metrics_ftn,
                      optimizer=optimizer,
                      config=config,
                      device=device,
                      dataloader_train=dataloader_train,
                      dataloader_valid=dataloader_valid,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch FaceParsing: Training')
    args.add_argument('--config', type=str, default=None, help='Path for the config file')
    args.add_argument('--resume', type=str, default=None, help='Path for the checkpoint file')
    args.add_argument('--device', type=str, default='0', help='Indices of GPUs to enable')
    args.add_argument('--run_id', type=str, default=None, help='Run ID')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='train_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)

    _main(config)
