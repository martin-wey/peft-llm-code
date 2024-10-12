#!/bin/bash

export HF_HOME="/Tmp/weyssowm/.cache/huggingface"

model="codellama/CodeLlama-7b-Instruct-hf"

CUDA_VISIBLE_DEVICES=2 python main.py \
  --model_name_or_path $model \
  --dataset apps \
  --tuning_method lora \
  --learning_rate 3e-4 \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --do_train \
  --use_wandb