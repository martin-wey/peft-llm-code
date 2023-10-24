#!/bin/bash

# LoRA -> array of tuples (lora_r, lora_alpha)
declare -a hps=(
  8 16
  8 32
  16 32
  16 64
  32 64
)

# In-context learning -> number of examples
n_shots=(0 1 2 3 4 5)

# Prompt-tuning -> number of virtual tokens
n_tokens=(5 10 20 40 100 200)

model=$1
training_method=$2
test_dataset=$3

if [ "$test_dataset" = "odex" ]; then
  prefix_dir="runs/test_code_generation_odex"
else
  prefix_dir="runs/test_code_generation_conala"
fi

if [ "${training_method}" = "icl" ]; then
  for n_shot in "${n_shots[@]}"; do
    echo "${model} - ${n_shot}-shot"

    python compute_metrics.py \
      --output_file "${prefix_dir}/${model}/output_${n_shot}shot.jsonl"
  done

elif [ "${training_method}" = "lora" ]; then
  for ((i = 0; i < ${#hps[@]}; i += 2)); do
    lora_r=${hps[i]}
    lora_alpha=${hps[i + 1]}

    run_name="${model}_lora_r${lora_r}a${lora_alpha}"
    echo $run_name

    python compute_metrics.py \
      --output_file "${prefix_dir}/${run_name}/output.jsonl"
  done

elif [ "${training_method}" = "prompt-tuning" ]; then
  for num_virtual_tokens in "${n_tokens[@]}"; do
    run_name="${model}_prompt${num_virtual_tokens}"
    echo $run_name

    python compute_metrics.py \
      --output_file "${prefix_dir}/${run_name}/output.jsonl"
  done
fi