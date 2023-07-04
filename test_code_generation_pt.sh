#!/bin/bash

n_tokens=(5 10 20 40 100 200)

model=$1
gpu_id=$2

model_name=$(echo "$model" | cut -d '/' -f 2)

for num_virtual_tokens in "${n_tokens[@]}"; do
  run_name="${model_name}_prompt${num_virtual_tokens}"
  output_dir="runs/test_code_generation/${run_name}/"

  if [[ ! -d "$output_dir" ]]; then
    mkdir -p "$output_dir"
  fi

  CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
    --task "code_generation" \
    --model_name_or_path $model \
    --training_method "prompt-tuning" \
    --adapter_path "runs/checkpoints/${run_name}/best_model_checkpoint" \
    --output_dir $output_dir \
    --do_test
done