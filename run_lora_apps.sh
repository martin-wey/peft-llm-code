#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python main.py \
  --model_name_or_path "codellama/CodeLlama-7b-Instruct-hf" \
  --dataset apps \
  --tuning_method qlora-4bit \
  --learning_rate 3e-4 \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --do_train \
  --use_wandb