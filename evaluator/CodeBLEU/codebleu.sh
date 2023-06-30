#!/bin/bash

model=$1

python calc_code_bleu.py \
  --refs "../../runs/test_code_generation/${model}/references.txt" \
  --hyp "../../runs/test_code_generation/${model}/predictions.txt" \
  --lang "python"
