#!/bin/bash
model=$1
seed=$2
output_dir=$3
gpu=$4

echo "GPU:", $gpu
echo "Output_dir:", $output_dir
echo "seed:", $seed
echo "model:", $model

CUDA_VISIBLE_DEVICES=${gpu} CUBLAS_WORKSPACE_CONFIG=:4096:8 python -W ignore -m heareval.predictions.runner ${output_dir}/hear_configs.${model}_r1_${seed}/* --random_seed ${seed}
