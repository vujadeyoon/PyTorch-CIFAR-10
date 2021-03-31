#!/bin/bash
#
#
path_config=$1
path_ckpt=$2
#
#
path_curr=$(pwd)
path_parent=$(dirname ${path_curr})
#
#
python3 ${path_curr}/main_test.py --device 0 --config ${path_curr}/${path_config} --resume ${path_curr}/${path_ckpt}