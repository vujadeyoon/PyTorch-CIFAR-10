import argparse
import collections
import torch
import dataloader.cifar_10 as dataloader_cifar_10
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from config.parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
from vujade import vujade_utils as utils_


def _main(config):
    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(is_cuda=config['trainer']['is_cuda'], n_gpu_use=config['n_gpu'])

    # fix random seeds for reproducibility
    utils_.set_seed(_device=device, _seed=config['trainer']['seed'])

    logger = config.get_logger('train')

    # setup data_loader instances
    dataloader_train = config.init_obj('train_loader', dataloader_cifar_10).loader
    dataloader_test = config.init_obj('test_loader', dataloader_cifar_10).loader

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])(_device=device)
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=dataloader_train,
                      valid_data_loader=dataloader_test,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Classification: CIFAR-10')
    args.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)

    _main(config)
