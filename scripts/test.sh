#!/usr/bin/env bash

script=${1:-'uvltrack'}
config=${2:-'baseline'}
dataset=${3:-'tnl2k'}
numgpu=${4:-2}
threads_per_gpu=${5:-8}

# CUDA_VISIBLE_DEVICES=2,3 \
nohup \
python tracking/test.py --tracker_name $script --tracker_param $config --dataset $dataset \
                        --threads $((threads_per_gpu*numgpu)) --num_gpus $numgpu --debug 0 \
> terminal_logs/test_$script'_'$config'_'$dataset.log 2>&1 &

echo log save to terminal_logs/test_$script'_'$config'_'$dataset.log