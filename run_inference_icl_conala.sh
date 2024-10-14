#!/bin/bash

models=(
  "codellama/CodeLlama-7b-hf"
  "codellama/CodeLlama-7b-Python-hf"
  "codellama/CodeLlama-7b-Instruct-hf"
  "Salesforce/codegen-350M-mono"
  "Salesforce/codegen2-1B_P"
  "Salesforce/codegen2-3_7B_P"
  "Salesforce/codegen2-7B_P"
  "Salesforce/codet5p-220m"
  "Salesforce/codet5p-770m"
)

ns=( 1 2 3 4 5 8 16 )

for model in "${models[@]}"; do
  for n in "${ns[@]}"; do
    echo "${model} - ICL n=${n}"
    CUDA_VISIBLE_DEVICES=0 python main.py \
      --model_name_or_path $model \
      --dataset "conala" \
      --num_rag_examples $n \
      --batch_size 1 \
      --do_test
  done
done