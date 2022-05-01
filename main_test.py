import os
import argparse
import torch
import dataloader.classification as module_dataloader
import metric.classification as module_metric
from model import model as module_arch
from tqdm import tqdm
from config.parse_config import ConfigParser
from vujade import vujade_path as path_
from vujade import vujade_imgcv as imgcv_
from vujade import vujade_segmentation as segm_
from vujade import vujade_str as str_
from vujade.vujade_debug import printf


def _main(config, _args) -> None:
    name_dataset = config['name']
    num_class = config['arch']['args']['num_classes']

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
    metric = config.init_obj('metric', module_metric, _num_class=num_class)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    if args.is_save is True:
        path_config = path_.Path(_spath=args.config)
        spath_result = os.path.join(path_config.replace(_old=config['trainer']['model_dir'].replace('.', ''), _new=config['visualization']['log_dir'].replace('.', '')).parent.str, 'result')
        path_result = path_.Path(_spath=spath_result)
        path_result.rmtree(_ignore_errors=True, _onerror=None)
        path_result.path.mkdir(mode=0o777, parents=True, exist_ok=True)
        printf('Path for saving result images: {}'.format(path_result.str), _is_pause=False)

    with torch.no_grad():
        for _idx, (_data, _target) in enumerate(tqdm(dataloader_test)):
            data, target = _data.to(device), _target.to(device)
            output = model(data)

            ndarr_mask_preds = output.cpu().numpy().argmax(1)
            ndarr_mask_targets = target.cpu().numpy()

            metric.add_batch(_predict=ndarr_mask_preds, _target=ndarr_mask_targets)

            if args.is_save is True:
                for _idy, (_filename, _datum, _ndarr_mask_pred, _ndarr_mask_target) in enumerate(zip(_filenames, _data, ndarr_mask_preds, ndarr_mask_targets)):
                    path_image_pred = path_.Path(_spath=os.path.join(path_result.str, _filename)).replace_ext(_new='_pred.png')
                    path_image_gt = path_.Path(_spath=os.path.join(path_result.str, _filename)).replace_ext(_new='_gt.png')

                    ndarr_cmask_pred = segm_.Visualize(_color_code=segm_.get_color_code(_name_dataset=name_dataset)).overlay(_ndarr_img=module_dataloader.inverse_preprocess(_input=_datum), _ndarr_mask=_ndarr_mask_pred)
                    ndarr_cmask_target = segm_.Visualize(_color_code=segm_.get_color_code(_name_dataset=name_dataset)).overlay(_ndarr_img=module_dataloader.inverse_preprocess(_input=_datum), _ndarr_mask=_ndarr_mask_target)
                    imgcv_.imwrite(_filename=path_image_pred.str, _ndarr=ndarr_cmask_pred, _is_rgb2bgr=False)
                    imgcv_.imwrite(_filename=path_image_gt.str, _ndarr=ndarr_cmask_target, _is_rgb2bgr=False)

    log = {
        'Run ID': _args.run_id,
        'Pixel Acc.': metric.overall_pixel_accuracy(),
        'Mean Acc.': metric.mean_pixel_accuracy(),
        'Mean IoU': metric.mean_iou(),
        'Mean F1-score': metric.mean_f1(),
        'IoU': metric.per_class_iou(_merge_ids=metric.get_merge_ids(_args=name_dataset)),
        'F1-score': metric.per_class_f1(_merge_ids=metric.get_merge_ids(_args=name_dataset))
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
