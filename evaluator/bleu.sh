#!/bin/bash

model=$1

python evaluator.py \
  --refs "../runs/conala_code_generation/${model}/best_model_checkpoint/references.txt" \
  --preds "../runs/conala_code_generation/${model}/best_model_checkpoint/predictions.txt"
