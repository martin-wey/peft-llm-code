#!/bin/bash

n_shots=(0 1 2 3 4 5)

model=$1
model_type=$2
gpu_id=$3
fp16=$4

model_name=$(echo "$model" | cut -d '/' -f 2)
path="runs/odex_pass_at_k_few_shot/${model_name}"

if [[ ! -d "$path" ]]; then
  mkdir -p "$path"
fi

for n_shot in "${n_shots[@]}"; do
  echo "${model} - ${n_shot}-shot"

  CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
    --task "odex_pass_at_k" \
    --model_name_or_path $model \
    --model_type $model_type \
    --output_dir $path \
    --num_few_shot_examples $n_shot \
    --num_beams 10 \
    --do_test \
    --fp16 $fp16
done
