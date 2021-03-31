#!/bin/bash
#
#
path_curr=$(pwd)
path_parent=$(dirname ${path_curr})
#
#
python3 ${path_curr}/main_train.py --device 0 --config ${path_curr}/config/cifar_10.json