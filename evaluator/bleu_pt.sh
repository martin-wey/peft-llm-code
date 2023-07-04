#!/bin/bash

model=$1
n_tokens=(5 10 20 40 100 200)

for num_virtual_tokens in "${n_tokens[@]}"; do
  run_name="${model}_prompt${num_virtual_tokens}"

  echo $run_name

  python evaluator.py \
    --refs "../runs/test_code_generation/${run_name}/references.txt" \
    --preds "../runs/test_code_generation/${run_name}/predictions.txt"
done
