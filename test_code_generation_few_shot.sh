#!/bin/bash

n_shots=(0 1 2 3 4 5)

model=$1
gpu_id=$2

model_name=$(echo "$model" | cut -d '/' -f 2)
output_dir="runs/test_code_generation/${model_name}"

if [[ ! -d "$output_dir" ]]; then
  mkdir -p "$output_dir"
fi

for n_shot in "${n_shots[@]}"; do
  echo "${model} - ${n_shot}-shot"

  CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
    --task "code_generation" \
    --model_name_or_path $model \
    --output_dir $output_dir \
    --num_few_shot_examples $n_shots \
    --fp16 True \
    --do_test
done
