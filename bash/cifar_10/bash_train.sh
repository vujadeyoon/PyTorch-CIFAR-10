#!/bin/bash
#
#
# Command:
#     i) Tensorboard.
#         1) Local
#             - tensorboard --logdir ./saved/
#         2) Remote
#             - tensorboard dev upload --logdir ./saved/ --name <NAME> --description <DESCRIPTION>
#             - tensorboard dev list
#             - tensorboard dev delete --experiment_id <EXPERIMENT_ID>
#     ii) Activate and deactivate to prevent the folder (e.g. ./saved/) from being deleted.
#         1) sudo chattr +a ./saved/
#         2) sudo chattr -a ./saved/
#     iii) Run bash script.
#         1) bash ./bash/cifar_10/bash_train.sh 0,1 exist_ok
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
PYTHONHASHSEED=0 python3 ${path_curr}/main_train.py --device ${device} \
                                                    --config ${path_curr}/config/cifar_10.yaml \
                                                    --run_id ${run_id}
#
#
# bash ./bash/cifar_10/bash_test.sh ${device} saved/model/cifar_10/${run_id}/config.yaml saved/model/cifar_10/${run_id}/ckpt-best.pth false false exist_ok
