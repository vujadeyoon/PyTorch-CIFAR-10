import os
import argparse
import torch
import dataloader.classification as module_dataloader
import loss.classification as module_loss
import metric.classification as module_metric
from model import model as module_arch
from utils import MetricTracker
from tqdm import tqdm
from config.parse_config import ConfigParser
from vujade import vujade_path as path_
from vujade import vujade_imgcv as imgcv_
from vujade import vujade_str as str_
from vujade.vujade_debug import printf


def _main(config, _args) -> None:
    logger = config.get_logger('test')

    # setup data_loader instances
    dataloader_test = config.init_obj('test_loader', module_dataloader).loader

    # build model architecture
    model = config.init_obj('arch', module_arch)
    if config['summary']['model']['is_print']:
        logger.info(model)
    logger.info('Loading checkpoint: {} ...'.format(config.resume))

    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get function handles of loss and metrics
    criterions = config.init_obj('loss', module_loss).to(device)
    metrics_ftn = [config.init_obj(('metric', met), module_metric) for met in config['metric']]
    metrics_test = MetricTracker('loss', 'lr', *[m.__class__.__name__ for m in metrics_ftn], writer=None)

    # prepare model for testing
    model = model.to(device)
    model.eval()
    metrics_test.reset()

    if args.is_save is True:
        raise NotImplementedError('This option has not been supported yet.')

    with torch.no_grad():
        for _idx, (_data, _target) in enumerate(tqdm(dataloader_test)):
            data, target = _data.to(device), _target.to(device)
            output = model(data)
            loss = criterions(output, target)

            # Update metric for iteration.
            metrics_test.update('loss', loss.item(), is_add_scalar=False)
            for met in metrics_ftn:
                metrics_test.update(met.__class__.__name__, met(output, target), is_add_scalar=False)

            if args.is_save is True:
                raise NotImplementedError('This option has not been supported yet.')

    log = {
        'Run ID': _args.run_id,
        'Loss': metrics_test.avg(key='loss'),
        'Accuracy': metrics_test.avg(key='Accuracy'),
        'AccuarcyTopK': metrics_test.avg(key='AccuarcyTopK')
    }

    logger.info(log)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='PyTorch FaceParsing: Testing')
    ap.add_argument('--config', type=str, required=True, help='Path for the config file')
    ap.add_argument('--resume', type=str, required=True, help='Path for the checkpoint file')
    ap.add_argument('--device', type=str, default='0', help='Indices of GPUs to enable')
    ap.add_argument('--is_log', type=str_.str2bool, default=False, help='Save log')
    ap.add_argument('--is_save', type=str_.str2bool, default=False, help='Save result image')
    ap.add_argument('--run_id', type=str, default=None, help='Run ID')
    args = ap.parse_args()

    config = ConfigParser.from_args(ap)

    _main(config, _args=args)

    if args.is_log is False:
        path_save_dir = path_.Path(_spath=str(config.save_dir))
        path_log_dir = path_.Path(_spath=str(config.log_dir))

        path_save_dir.rmtree(_ignore_errors=True, _onerror=None)
        path_log_dir.rmtree(_ignore_errors=True, _onerror=None)
