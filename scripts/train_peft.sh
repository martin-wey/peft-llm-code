#!/bin/bash

pefts_lrs=("lora,3e-4" "ia3,3e-3" "prompt-tuning,3e-3" "prefix-tuning,3e-2")

model=$1
model_name=$(echo "$model" | rev | cut -d'/' -f1 | rev)
dataset=$2
batch_size=$3
gradient_accumulation_steps=$4
gpu_id=$5

for tuple in "${pefts_lrs[@]}"; do
  IFS=',' read -ra values <<< "$tuple"
  peft="${values[0]}"
  lr=${values[1]}

  echo "${model_name} - ${peft} - ${dataset}"
  CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
    --model_name_or_path $model \
    --dataset $dataset \
    --tuning_method $peft \
    --learning_rate $lr \
    --batch_size $batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --do_train \
    --use_wandb
done
