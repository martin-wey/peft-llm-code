#!/bin/bash

# array of tuples (lora_r, lora_alpha)
declare -a hps=(
  8 16
  8 32
  16 32
  16 64
  32 64
)

model_name=$1

for ((i = 0; i < ${#hps[@]}; i += 2)); do
  lora_r=${hps[i]}
  lora_alpha=${hps[i + 1]}

  run_name="${model_name}_lora_r${lora_r}a${lora_alpha}"

  echo $run_name

  python evaluator.py \
    --refs "../runs/test_code_generation/${run_name}/references.txt" \
    --preds "../runs/test_code_generation/${run_name}/predictions.txt"
done