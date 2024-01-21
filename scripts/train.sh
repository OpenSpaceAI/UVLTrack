#!/usr/bin/env bash

script=${1:-'uvltrack'}
config=${2:-'baseline_base'}
numgpu=${3:-2}
gpuid=${4:-'0,1'}

CUDA_VISIBLE_DEVICES=$gpuid \
nohup \
python tracking/train.py --script $script \
                        --config $config \
                        --save_dir . \
                        --mode multiple \
                        --nproc_per_node $numgpu \
> terminal_logs/train_$script'_'$config.log 2>&1 &

echo log save to terminal_logs/train_$script'_'$config.log