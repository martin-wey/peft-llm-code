#!/bin/bash

n_shots=(0 1 2 3 4 5)

model=$1
model_type=$2
gpu_id=$3
fp16=$4

model_name=$(echo "$model" | cut -d '/' -f 2)
output_dir="runs/test_pass_at_k/${model_name}"

if [[ ! -d "$output_dir" ]]; then
  mkdir -p "$output_dir"
fi

for n_shot in "${n_shots[@]}"; do
  echo "${model} - ${n_shot}-shot"

  CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
    --task "pass_at_k" \
    --model_name_or_path $model \
    --model_type $model_type \
    --output_dir $output_dir \
    --num_few_shot_examples $n_shot \
    --num_beams 10 \
    --do_test \
    --fp16 $fp16
done
