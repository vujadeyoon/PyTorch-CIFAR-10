import os
import argparse
import collections
import torch
from model import mobileone as module_arch
from config.parse_config import ConfigParser
from vujade import vujade_path as path_
from vujade import vujade_str as str_
from vujade import vujade_torch as torch_
from vujade.vujade_debug import printd


def _main(config) -> None:
    path_resume = path_.Path(str(config.resume))
    path_model_entire = path_.Path(os.path.join(path_resume.parent.str, 'model_entire{}'.format(path_resume.ext)))
    path_model_traced = path_.Path(os.path.join(path_resume.parent.str, 'model_traced{}'.format(path_resume.ext)))
    path_model_scripted = path_.Path(os.path.join(path_resume.parent.str, 'model_scripted{}'.format(path_resume.ext)))

    device, gpu_ids = torch_.PyTorchUtils.prepare_device(_is_cuda=False)

    # build model architecture
    model = config.init_obj('arch', module_arch)

    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint['state_dict']

    if 1 < len(gpu_ids): # Multi-GPUs
        is_jit = False
        model = torch.nn.DataParallel(model)
    else: # CPU or single-GPU
        is_jit = True
        state_dict = torch_.PyTorchUtils.convert_ckpt(_state_dict=state_dict, _device=device)

    # Load checkpoint
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    # Save entire model
    torch.save(model, path_model_entire.str)

    if is_jit is True:
        # Export entire model in TorchScript format
        model_trace = torch.jit.trace(model, torch.rand((1, *config['test_loader']['args']['size'])).to(device))
        model_trace.save(path_model_traced.str)

        model_scripted = torch.jit.script(model)
        model_scripted.save(path_model_scripted.str)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='PyTorch Classification: Jit')
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
