#!/bin/bash
#
#
device=$1
path_config=$2
path_ckpt=$3
is_log=${4:-false}
run_id=${5:-$(date +%y)$(date +%m)$(date +%d)_$(date +%H)$(date +%M)$(date +%S)}
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
fi
#
#
python3 ${path_curr}/main_test.py --config ${path_config} \
                                  --resume ${path_ckpt} \
                                  --is_log ${is_log} \
                                  --run_id ${run_id}
