#!/bin/bash
#
#
path_code=$(pwd)
path_base=/DATA/Dataset
name_dataset=CIFAR_10
path_dataset=${path_base}/${name_dataset}/
url_cifar=https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
path_cifar=${path_dataset}/cifar-10-batches-py
name_targz=cifar.tar.gz
path_temp=${path_dataset}/temp
#
#
unset PYTHONPATH
export PYTHONPATH=${PYTHONPATH}:${path_code}
#
#
mkdir -p ${path_dataset}
#
#
cd ${path_dataset} && wget -O ${name_targz} ${url_cifar} && tar -xzvf ${name_targz} && rm -f ${name_targz} && mv ${path_cifar} ${path_temp}
#
#
cd ${path_code}/dataset && python3 cifar_10.py --name_dataset ${name_dataset} --path_dataset ${path_dataset} && rm -rf ${path_temp}
#
#
cd ${path_code} || return
