#!/bin/bash

model=$1

python evaluator.py \
  --refs "../runs/test_code_generation/${model}/references.txt" \
  --preds "../runs/test_code_generation/${model}/predictions.txt"
