#!/bin/bash

n_tokens=(5 10 20 40 100 200)

model=$1
model_type=$2
gpu_id=$3

model_name=$(echo "$model" | cut -d '/' -f 2)

for num_virtual_tokens in "${n_tokens[@]}"; do
  run_name="${model_name}_pt${num_virtual_tokens}"
  output_dir="runs/test_pass_at_k/${run_name}"

  if [[ ! -d "$output_dir" ]]; then
    mkdir -p "$output_dir"
  fi

  CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
    --task "pass_at_k" \
    --model_name_or_path $model \
    --model_type $model_type \
    --training_method "prefix-tuning" \
    --adapter_path "runs/checkpoints/${run_name}/best_model_checkpoint" \
    --output_dir $output_dir \
    --num_beams 10 \
    --do_test
done