#!/bin/bash

model=$1
n_shots=(0 1 2 3 4 5)

for n_shot in "${n_shots[@]}"; do
  echo "$model - $n_shot"
  python calc_code_bleu.py \
    --refs "../../runs/test_code_generation/${model}/references_${n_shot}shot.txt" \
    --hyp "../../runs/test_code_generation/${model}/predictions_${n_shot}shot.txt" \
    --lang "python"
done
