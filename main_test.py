import argparse
import collections
import torch
import dataloader.classification as module_dataloader
import loss.classification as module_loss
import metric.classification as module_metric
from model import model as module_arch
from utils import MetricTracker
from tqdm import tqdm
from config.parse_config import ConfigParser
from vujade import vujade_path as path_
from vujade import vujade_str as str_
from vujade import vujade_torch as torch_
from vujade.vujade_debug import printd


def _main(config) -> None:
    logger = config.get_logger('test')

    device, gpu_ids = torch_.PyTorchUtils.prepare_device(_is_cuda=config['trainer']['is_cuda'])

    # setup data_loader instances
    dataloader_test = config.init_obj('test_loader', module_dataloader).loader

    # build model architecture
    model = config.init_obj('arch', module_arch)
    if config['summary']['model']['is_print']:
        logger.info(model)
    logger.info('Loading checkpoint: {} ...'.format(config.resume))

    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if 1 < len(gpu_ids):
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # get function handles of loss and metrics
    criterions = config.init_obj('loss', module_loss).to(device)
    metrics_ftn = [config.init_obj(('metric', met), module_metric) for met in config['metric']]
    metrics_test = MetricTracker('loss', 'lr', *[m.__class__.__name__ for m in metrics_ftn], writer=None)

    # prepare model for testing
    model = model.to(device)
    model.eval()
    metrics_test.reset()

    with torch.no_grad():
        for _idx, (_data, _target) in enumerate(tqdm(dataloader_test)):
            data, target = _data.to(device), _target.to(device)
            output = model(data)
            loss = criterions(output, target)

            # Update metric for iteration.
            metrics_test.update('loss', loss.item(), is_add_scalar=False)
            for met in metrics_ftn:
                metrics_test.update(met.__class__.__name__, met(output, target), is_add_scalar=False)

    log = {
        'Run ID': config['run_id'],
        'Loss': metrics_test.avg(key='loss'),
        'Accuracy': metrics_test.avg(key='Accuracy'),
        'AccuarcyTopK': metrics_test.avg(key='AccuarcyTopK')
    }

    logger.info(log)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='PyTorch Classification: Testing')
    ap.add_argument('--config', type=str, required=True, help='Path for the config file')
    ap.add_argument('--resume', type=str, required=True, help='Path for the checkpoint file')
    ap.add_argument('--is_log', type=str_.str2bool, default=False, help='Save log')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--run_id'], type=str, target='run_id')
    ]

    config, args = ConfigParser.from_args(ap, options), ap.parse_args()

    _main(config)

    if args.is_log is False:
        path_save_dir = path_.Path(_spath=str(config.save_dir))
        path_log_dir = path_.Path(_spath=str(config.log_dir))

        path_save_dir.rmtree(_ignore_errors=True, _onerror=None)
        path_log_dir.rmtree(_ignore_errors=True, _onerror=None)
