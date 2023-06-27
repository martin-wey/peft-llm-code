#!/bin/bash

# array of tuples (lora_r, lora_alpha)
declare -a hps=(
  8 16
  8 32
  16 32
  16 64
  32 64
  64 128
  128 256
)

model=$1
model_type=$2
batch_size=$3
gradient_accumulation_steps=$4
gpu_id=$5
fp16=$6

model_name=$(echo "$model" | cut -d '/' -f 2)

for ((i = 0; i < ${#hps[@]}; i += 2)); do
  lora_r=${hps[i]}
  lora_alpha=${hps[i + 1]}

  run_name="${model_name}_lora_r${lora_r}a${lora_alpha}"

  CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
    --model_name_or_path $model \
    --training_method "lora" \
    --learning_rate 3e-4 \
    --run_name $run_name \
    --lora_r $lora_r \
    --lora_alpha $lora_alpha \
    --batch_size $batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --gpu_id $gpu_id \
    --fp16 $fp16 \
    --do_train \
    --use_wandb
done