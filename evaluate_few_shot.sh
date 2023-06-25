#!/bin/bash

# array of tuples (n_shots, conala_max_input_length)
declare -a hps=(
  0 64
  1 128
  2 256
  3 512
  4 512
  5 512
)

model=$1
model_type=$2
batch_size=$3
gpu_id=$4
fp16=$5

model_name=$(echo "$model" | cut -d '/' -f 2)
path="runs/conala_few_shot/${model_name}"

if [[ ! -d "$path" ]]; then
  mkdir -p "$path"
fi

for ((i = 0; i < ${#hps[@]}; i += 2)); do
  n_shots=${hps[i]}
  max_len=${hps[i + 1]}
  echo "${model} - ${n_shots}-shot - ${max_len}"

  CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
    --task "conala_code_generation" \
    --model_name_or_path $model \
    --model_type $model_type \
    --output_dir $path \
    --num_few_shot_examples $n_shots \
    --conala_max_input_length $max_len \
    --batch_size $batch_size \
    --do_test \
    --fp16 $fp16
done
