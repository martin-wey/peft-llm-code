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
gpu_id=$3

model_name=$(echo "$model" | cut -d '/' -f 2)

for ((i = 0; i < ${#hps[@]}; i += 2)); do
  lora_r=${hps[i]}
  lora_alpha=${hps[i + 1]}

  run_name="${model_name}_lora_r${lora_r}a${lora_alpha}"
  output_dir="runs/test_pass_at_k/${run_name}"

  if [[ ! -d "$output_dir" ]]; then
    mkdir -p "$output_dir"
  fi

  CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
    --task "pass_at_k" \
    --model_name_or_path $model \
    --model_type $model_type \
    --training_method "lora" \
    --adapter_path "runs/checkpoints/${run_name}/best_model_checkpoint" \
    --output_dir $output_dir \
    --num_beams 10 \
    --do_test
done