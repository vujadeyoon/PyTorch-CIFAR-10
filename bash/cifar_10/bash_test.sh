#!/bin/bash
#
#
# Command: bash ./bash/cifar_10/bash_test.sh 2,3 saved/model/cifar_10/220104_001830/config.yaml saved/model/cifar_10/220104_001830/ckpt-best.pth false false exist_ok
#
#
device=${1:-0,1}
path_config=$2
path_ckpt=$3
is_log=${4:-false}
is_save=${5:-false}
run_id=${6:-$(date +%y)$(date +%m)$(date +%d)_$(date +%H)$(date +%M)$(date +%S)}
#
#
path_curr=$(pwd)
path_parent=$(dirname ${path_curr})
#
#
python3 ${path_curr}/main_test.py --device ${device} \
                                  --config ${path_curr}/${path_config} \
                                  --resume ${path_curr}/${path_ckpt} \
                                  --is_log ${is_log} \
                                  --is_save ${is_save} \
                                  --run_id ${run_id}
