#!/usr/bin/env bash

args=("$@")

joined_args=$(printf "%s_" "${args[@]}")
joined_args=${joined_args%_}

safe_args="${joined_args// /_}"

run_dir="phi-3-mini-128k-instruct_codealpaca_qlora8b_${safe_args}"
run_name="args_${safe_args}"

CUDA_VISIBLE_DEVICES=0 python sft.py \
  --output_dir "runs/${run_dir}" \
  --model_name_or_path "microsoft/Phi-3-mini-128k-instruct" \
  --trust_remote_code \
  --dataset_name datasets/codealpaca \
  --torch_dtype bfloat16 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 3 \
  --learning_rate 3e-4 \
  --max_seq_length 512 \
  --gradient_checkpointing \
  --optim adafactor \
  --use_peft \
  --use_lora \
  --load_in_8bit \
  --lora_target_modules o_proj, qkv_proj \
  --completion_only \
  --response_template "<|assistant|>" \
  --eval_strategy "epoch" \
  --logging_steps 1 \
  --save_strategy "epoch" \
  --report_to "wandb" \
  --run_name "${run_name}" \
  "${args[@]}"
