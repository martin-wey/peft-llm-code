#!/bin/bash

model=$1
model_type=$2
gpu_id=$3

checkpoint="runs/checkpoints/${model}_ft/best_model_checkpoint"
output_dir="runs/test_pass_at_k/${model}_ft"

if [[ ! -d "$output_dir" ]]; then
  mkdir -p "$output_dir"
fi

CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
  --task "pass_at_k" \
  --model_name_or_path $checkpoint \
  --model_type $model_type \
  --output_dir $output_dir \
  --num_beams 10 \
  --do_test