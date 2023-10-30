#!/bin/bash

pefts=("lora" "ia3" "prompt-tuning" "prefix-tuning")

model=$1
model_name=$(echo "$model" | rev | cut -d'/' -f1 | rev)
dataset=$2
gpu_id=$3

for peft in "${pefts[@]}"; do
  adapter_path="runs/checkpoints/${dataset}/${model_name}_${peft}"
  echo "${model_name} - ${peft} - ${dataset}"
  CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
  --model_name_or_path $model \
  --adapter_path $adapter_path \
  --tuning_method $peft \
  --dataset $dataset \
  --do_test
done
