import argparse
import torch
import numpy as np
import torch.nn.functional as F
from vujade import vujade_imgcv as imgcv_
from vujade import vujade_path as path_
from vujade import vujade_str as str_
from vujade import vujade_torch as torch_
from vujade import vujade_transforms as trans_
from vujade.vujade_debug import printd


def _main(_args) -> None:
    path_img = path_.Path(_args.path_img)
    path_pth = path_.Path(_args.path_pth)

    if path_img.path.is_file() is False:
        raise FileNotFoundError('The input image, {} is not existed.'.format(path_img.str))

    if path_pth.path.is_file() is False:
        raise FileNotFoundError('The trained model, {} is not existed.'.format(path_pth.str))

    device, gpu_ids = torch_.PyTorchUtils.prepare_device(_is_cuda=_args.is_cuda)

    if _args.is_jit is True:
        model = torch.jit.load(path_pth.str).to(device).eval()
    else:
        model = torch.load(path_pth.str).to(device).eval()

    ndarr_img_ori = imgcv_.imread(_filename=path_img.str).astype(np.float32)
    tensor_input = trans_.ndarr2tensor(_ndarr=trans_.Standardize.forward(_ndarr=ndarr_img_ori, _mean=_args.mean, _std=_args.std).astype(np.float32)).to(device)

    with torch.no_grad():
        tensor_output = model(tensor_input)
        result = int(torch.argmax(F.softmax(tensor_output, dim=1), dim=1, keepdim=False).detach().cpu().numpy())

    printd('tensor_output: ', type(tensor_output), tensor_output.device, tensor_output.dtype, tensor_output.shape, tensor_output, _is_pause=False)
    printd('result: ', type(result), result, _is_pause=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='PyTorch Classification: Jit')
    ap.add_argument('--path_pth', type=str, required=True, help='Path for the trained model')
    ap.add_argument('--path_img', type=str, default='./asset/image/abandoned_ship_s_000213.png', help='Path for an input image')
    ap.add_argument('--is_cuda', type=str_.str2bool, default=False, help='Boolean variable for CUDA')
    ap.add_argument('--is_jit', type=str_.str2bool, default=True, help='Boolean variable for torch.jit')
    ap.add_argument('--mean', nargs='+', default=[0.4914, 0.4822, 0.4465], help='List for mean')
    ap.add_argument('--std', nargs='+', default=[0.2023, 0.1994, 0.2010], help='List for std')
    args = ap.parse_args()

    # Update list arguments.
    args.mean = str_.str2list_v2(_str=args.mean[0])
    args.std = str_.str2list_v2(_str=args.std[0])

    _main(_args=args)
