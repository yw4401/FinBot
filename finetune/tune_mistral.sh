#!/bin/bash
gsutil cp gs://scraped-news-article-data-null/fine-tune-summary-train.parquet .
gsutil cp gs://scraped-news-article-data-null/fine-tune-summary-test.parquet .
deepspeed tune_summary.py --deepspeed=deepspeed_llama2.json --output_dir="./summary-mistral-lora" \
 --per_device_train_batch_size=1 --learning_rate=1e-4 --num_train_epochs=4 --weight_decay=0.1 --warmup_steps=200 \
 --bf16=True --lr_scheduler_type="cosine" --model_path="Open-Orca/Mistral-7B-OpenOrca" --remove_unused_columns=False \
 --sample=1000000 --save_strategy "epoch" --lora_target "q_proj,k_proj,v_proj,o_proj" --lora_r 16 --lora_alpha 16 \
 --gradient_checkpointing True --evaluation_strategy="epoch" --logging_strategy="epoch" --per_device_eval_batch_size=4 \
 --generation_max_length 256
