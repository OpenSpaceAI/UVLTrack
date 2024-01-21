#!/usr/bin/env bash


script=$1
config=$2
input_video=$3 # <input video path>
output_video=$4 # <output video path>
language=${5:""} # <language description of target>
init_bbox=${6:""} # <initial bbox of target: x y w h>

python demo.py  --tracker_name $script  \
                --tracker_param $config  \
                --input_video $input_video \
                --output_video $output_video \
                --language $language \
                --init_bbox $init_bbox \