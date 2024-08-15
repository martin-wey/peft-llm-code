#!/usr/bin/env bash

args=("$@")

joined_args=$(printf "%s_" "${args[@]}")
joined_args=${joined_args%_}

safe_args="${joined_args// /_}"

run_name="phi-3-mini-128k-instruct_lora_${safe_args}"

CUDA_VISIBLE_DEVICES=0 python sft.py \
  --output_dir "runs/${run_name}" \
  --model_name_or_path "Microsoft/Phi-3-mini-128k-instruct" \
  --trust_remote_code \
  --dataset_name datasets/conala \
  --torch_dtype bfloat16 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 3 \
  --learning_rate 3e-4 \
  --max_seq_length 128 \
  --gradient_checkpointing \
  --optim adafactor \
  --use_peft \
  --lora_target_modules o_proj, qkv_proj \
  --completion_only \
  --response_template "<|assistant|>" \
  --eval_strategy "epoch" \
  --logging_steps 1 \
  --save_strategy "epoch" \
  --report_to "wandb" \
  --run_name "${run_name}" \
  "${args[@]}"
