#!/bin/bash

n_tokens=(5 10 20 40 100 200)

model=$1
model_type=$2
batch_size=$3
gradient_accumulation_steps=$4
gpu_id=$5
fp16=$6

model_name=$(echo "$model" | cut -d '/' -f 2)

for num_virtual_tokens in "${n_tokens[@]}"; do
  echo "${model} - ${num_virtual_tokens} virtual tokens"

  run_name="${model_name}_prompt${num_virtual_tokens}"

  CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
    --model_name_or_path $model \
    --training_method "prompt-tuning" \
    --num_virtual_tokens $num_virtual_tokens \
    --learning_rate 3e-3 \
    --run_name $run_name \
    --batch_size $batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --gpu_id $gpu_id \
    --fp16 $fp16 \
    --do_train \
    --use_wandb
done