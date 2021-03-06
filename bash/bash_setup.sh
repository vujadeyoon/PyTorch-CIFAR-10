#!/bin/bash
#
#
# Command: bash ./bash/bash_setup.sh
#
#
path_curr=$(pwd)
path_parent=$(dirname ${path_curr})
path_cython_utils=${path_curr}/vujade/utils
path_cython_distance=${path_cython_utils}/Distance
path_cython_nms=${path_cython_utils}/NMS/cython_nms
path_cython_scd_batch=${path_cython_utils}/SceneChangeDetection/BatchProcessing
path_cython_scd_inter=${path_cython_utils}/SceneChangeDetection/InteractiveProcessing
#
#
find ${path_curr}/ -name __pycache__ -exec rm -rf {} \;
find ${path_curr}/ -name .idea -exec rm -rf {} \;
#
#
cd ${path_cython_distance} && rm -rf build ./*distance*.c ./*distance*.so
cd ${path_cython_nms} && rm -rf build ./*nms*.c ./*nms*.so
cd ${path_cython_scd_batch} && rm -rf build ./*scd*.c ./*scd*.so
cd ${path_cython_scd_inter} && rm -rf build ./*scd*.c ./*scd*.so
#
#
cd ${path_cython_distance} && python3 setup.py build_ext --inplace
cd ${path_cython_nms} && python3 setup.py build_ext --inplace
cd ${path_cython_scd_batch} && python3 setup.py build_ext --inplace
cd ${path_cython_scd_inter} && python3 setup.py build_ext --inplace