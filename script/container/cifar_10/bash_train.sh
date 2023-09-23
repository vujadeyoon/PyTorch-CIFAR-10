#!/bin/bash
#
#
device=${1:-0,1}
run_id=${2:-$(date +%y)$(date +%m)$(date +%d)_$(date +%H)$(date +%M)$(date +%S)}
#
#
path_curr=$(pwd)
path_parent=$(dirname ${path_curr})
#
#
unset PYTHONPATH PYTHONDONTWRITEBYTECODE PYTHONUNBUFFERED PYTHONHASHSEED CUDA_VISIBLE_DEVICES SLACK_TOKEN_USR SLACK_TOKEN_BOT
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
export SLACK_TOKEN_USR='SECRET_SLACK_TOKEN_USR'
export SLACK_TOKEN_BOT='SECRET_SLACK_TOKEN_BOT'
#
#
time_train_start=$(date +%s)
#
#
python3 ${path_curr}/main_train.py --config ${path_curr}/config/cifar_10.yaml \
                                   --run_id ${run_id}
#
#
time_train_end=$(date +%s)
#
#
bash ${path_curr}/script/cifar_10/bash_test.sh ${device} \
                                               ${path_curr}/saved/model/CIFAR_10/${run_id}/config.yaml \
                                               ${path_curr}/saved/model/CIFAR_10/${run_id}/ckpt-best.pth \
                                               false \
                                               exist_ok_test
#
#
echo "Elapsed time for training [sec.]: $((${time_train_end}-${time_train_start}))"
