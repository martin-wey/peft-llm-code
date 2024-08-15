#!/bin/bash

n_examples=(1 2 3)
seeds=(42 777 5432 55555 97)

model=$1
model_name=$(echo "$model" | rev | cut -d'/' -f1 | rev)
dataset=$2
gpu_id=$3

for n in "${n_examples[@]}"; do
  for seed in "${seeds[@]}"; do
    run_name="${model_name}_icl_seed${seed}"
    echo "inference - ${run_name}"
    CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
      --model_name_or_path $model \
      --run_name $run_name \
      --dataset $dataset \
      --num_icl_examples $n \
      --do_test \
      --seed $seed
  done
done
