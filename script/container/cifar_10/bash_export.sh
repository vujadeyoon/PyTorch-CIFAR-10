#!/bin/bash
#
#
path_config=$1
path_ckpt=$2
is_log=${3:-false}
run_id=${4:-$(date +%y)$(date +%m)$(date +%d)_$(date +%H)$(date +%M)$(date +%S)}
#
#
path_curr=$(pwd)
path_parent=$(dirname ${path_curr})
#
#
unset PYTHONPATH PYTHONDONTWRITEBYTECODE PYTHONUNBUFFERED PYTHONHASHSEED
export PYTHONPATH=${PYTHONPATH}:${path_curr}
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
#
#
python3 ${path_curr}/main_export.py --config ${path_config} \
                                    --resume ${path_ckpt} \
                                    --is_log ${is_log} \
                                    --run_id ${run_id}
