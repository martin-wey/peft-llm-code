#!/bin/bash

# array of tuples (lora_r, lora_alpha)
declare -a hps=(
  8 16
  8 32
  16 32
  16 64
  32 64
)

model=$1
model_type=$2
batch_size=$3
gpu_id=$4

model_name=$(echo "$model" | cut -d '/' -f 2)

for ((i = 0; i < ${#hps[@]}; i += 2)); do
  lora_r=${hps[i]}
  lora_alpha=${hps[i + 1]}

  run_name="${model_name}_lora_r${lora_r}a${lora_alpha}"

  CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
    --model_name_or_path $model \
    --training_method "lora" \
    --adapter_path "runs/conala_code_generation/${run_name}/best_model_checkpoint" \
    --output_dir "runs/conala_code_generation/${run_name}/best_model_checkpoint" \
    --batch_size $batch_size \
    --do_test
done