#!/bin/bash

model=$1

python calc_code_bleu.py \
  --refs "../../runs/conala_code_generation/${model}/best_model_checkpoint/references.txt" \
  --hyp "../../runs/conala_code_generation/${model}/best_model_checkpoint/predictions.txt" \
  --lang "python"
