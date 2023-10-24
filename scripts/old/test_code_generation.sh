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
gpu_id=$4

if [ "$test_dataset" = "odex" ]; then
  output_prefix_dir="runs/test_code_generation_odex"
else
  output_prefix_dir="runs/test_code_generation_conala"
fi

model_name=$(echo "$model" | cut -d '/' -f 2)

if [ "${training_method}" = "ft" ]; then
  output_dir="${output_prefix_dir}/${model_name}"
  checkpoint="runs/checkpoints/${model}/best_model_checkpoint"

  if [[ ! -d "$output_dir" ]]; then
    mkdir -p "$output_dir"
  fi

  CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
    --test_dataset $test_dataset \
    --model_name_or_path $checkpoint \
    --output_dir $output_dir \
    --do_test

elif [ "${training_method}" = "icl" ]; then
  for n_shot in "${n_shots[@]}"; do
    echo "${model} - ${n_shot}-shot"
    CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
      --test_dataset $test_dataset \
      --model_name_or_path $model \
      --output_dir $output_dir \
      --num_few_shot_examples $n_shot \
      --fp16 True \
      --do_test
  done

elif [ "${training_method}" = "lora" ]; then
  for ((i = 0; i < ${#hps[@]}; i += 2)); do
    lora_r=${hps[i]}
    lora_alpha=${hps[i + 1]}

    run_name="${model_name}_lora_r${lora_r}a${lora_alpha}"
    output_dir="${prefix_dir}/${run_name}/"
    checkpoint="runs/checkpoints/${run_name}/best_model_checkpoint"

    if [[ ! -d "$output_dir" ]]; then
      mkdir -p "$output_dir"
    fi

    echo $run_name
    CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
      --test_dataset $test_dataset \
      --model_name_or_path $model \
      --training_method "lora" \
      --adapter_path $checkpoint \
      --output_dir $output_dir \
      --do_test
  done

elif [ "${training_method}" = "prompt-tuning" ]; then
  for num_virtual_tokens in "${n_tokens[@]}"; do
    run_name="${model_name}_prompt${num_virtual_tokens}"
    output_dir="${prefix_dir}/${run_name}/"
    checkpoint="runs/checkpoints/${run_name}/best_model_checkpoint"

    if [[ ! -d "$output_dir" ]]; then
      mkdir -p "$output_dir"
    fi

    echo $run_name
    CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
      --test_dataset $test_dataset \
      --model_name_or_path $model \
      --training_method "prompt-tuning" \
      --adapter_path $checkpoint \
      --output_dir $output_dir \
      --do_test
  done
fi