import os
import argparse
import collections
import torch
import dataloader.classification as module_dataloader
import loss.classification as module_loss
import metric.classification as module_metric
from model import repvit as module_arch
from config.parse_config import ConfigParser
from trainer import LRScheduler, Trainer
from vujade import vujade_path as path_
from vujade import vujade_torch as torch_
from vujade import vujade_utils as utils_
from vujade.vujade_debug import printd


def backup(_cmd: str, _spath_backup: str) -> bool:
    is_success, _ = utils_.SystemCommand.run(_command=_cmd, _is_daemon=False, _is_subprocess=True)
    path_backup = path_.Path(_spath=_spath_backup)
    if (is_success & path_backup.path.is_file()) is True:
        res = True
    else:
        res = False

    return res


def _main(config):
    logger = config.get_logger('train')

    if config['backup']['is_backup'] is True:
        spath_backup = config['backup']['cmd'].split()[2]
        if backup(_cmd=config['backup']['cmd'], _spath_backup=spath_backup):
            logger.info('The backup is successful to {}.'.format(spath_backup))
        else:
            logger.info('The backup failed to {}.'.format(spath_backup))

    device, gpu_ids = torch_.PyTorchUtils.prepare_device(_is_cuda=config['trainer']['is_cuda'])

    # disable AMP when running on CPU
    if device == torch.device('cpu'):
        config['trainer']['is_amp'] = False

    # fix random seeds for reproducibility
    utils_.SetSeed.fix_seed(
        _device=device,
        _seed=config['trainer']['seed'],
        _is_use_deterministic_algorithm=config['trainer']['is_use_deterministic_algorithm']
    )

    # setup data_loader instances
    dataloader_train = config.init_obj('train_loader', module_dataloader).loader
    dataloader_valid = config.init_obj('valid_loader', module_dataloader).loader

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)

    if config['summary']['model']['is_print']:
        logger.info(model)

    if config['summary']['computational_cost']['is_print']:
        macs, flops, params = torch_.CalculateComputationalCost(
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
    if 1 < len(gpu_ids):
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    # get function handles of loss and metrics
    criterions = config.init_obj('loss', module_loss).to(device)
    metrics_ftn = [config.init_obj(('metric', met), module_metric) for met in config['metric']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = LRScheduler(
        _config=config,
        _optimizer=optimizer,
        _warmup_alogrithm=config['lr_scheduler']['warmup_alogrithm'],
        _warmup_epoch=config['lr_scheduler']['warmup_epoch'],
        _warmup_lr=config['lr_scheduler']['warmup_lr']
    )

    trainer = Trainer(
        model=model,
        criterions=criterions,
        metrics_ftn=metrics_ftn,
        optimizer=optimizer,
        config=config,
        device=device,
        dataloader_train=dataloader_train,
        dataloader_valid=dataloader_valid,
        lr_scheduler=lr_scheduler
    )

    trainer.train()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='PyTorch Classification: Training')
    ap.add_argument('--config', type=str, default=None, help='Path for the config file')
    ap.add_argument('--resume', type=str, default=None, help='Path for the checkpoint file')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--run_id'], type=str, target='run_id')
    ]

    config, args = ConfigParser.from_args(ap, options), ap.parse_args()

    _main(config)
