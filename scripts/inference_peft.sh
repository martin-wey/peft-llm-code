#!/bin/bash

pefts=("lora" "ia3" "prompt-tuning" "prefix-tuning")

model=$1
model_name=$(echo "$model" | rev | cut -d'/' -f1 | rev)
dataset=$2
gpu_id=$3

for peft in "${pefts[@]}"; do
  peft="${values[0]}"
  lr=${values[1]}

  echo "${model_name} - ${peft} - ${dataset}"
  CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
  --model_name_or_path $model \
  --run_name $run_name \
  --dataset $dataset \
  --num_icl_examples 0 \
  --do_test
done
