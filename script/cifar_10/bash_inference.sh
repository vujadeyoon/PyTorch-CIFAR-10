#!/bin/bash
#
#
device=$1
is_jit=$2
path_pth=$3
path_img=${4:-./asset/image/abandoned_ship_s_000213.png}
mean="${5:-[0.4914, 0.4822, 0.4465]}"
std="${6:-[0.2023, 0.1994, 0.2010]}"
#
#
path_curr=$(pwd)
path_parent=$(dirname ${path_curr})
#
#
unset PYTHONPATH PYTHONDONTWRITEBYTECODE PYTHONUNBUFFERED PYTHONHASHSEED CUDA_VISIBLE_DEVICES
export PYTHONPATH=${PYTHONPATH}:${path_curr}
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
#
#
if [[ ${device} == *","* ]] || [[ ${device} -ge "0" ]]; then
  export CUDA_VISIBLE_DEVICES=${device}
  is_cuda=true
else
  is_cuda=false
fi
#
#
python3 ${path_curr}/main_inference.py --path_pth ${path_pth} \
                                       --path_img ${path_img} \
                                       --is_cuda ${is_cuda} \
                                       --is_jit ${is_jit} \
                                       --mean "${mean}" \
                                       --std "${std}"
