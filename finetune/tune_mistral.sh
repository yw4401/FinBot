#!/bin/bash
gsutil cp gs://scraped-news-article-data-null/fine-tune-summary-train.parquet .
gsutil cp gs://scraped-news-article-data-null/fine-tune-summary-test.parquet .
deepspeed llama_summary.py --deepspeed=deepspeed_llama2.json --output_dir="./summary-mistral-lora" \
 --per_device_train_batch_size=1 --learning_rate=1e-4 --num_train_epochs=1 --weight_decay=0.1 --warmup_steps=200 \
 --bf16=True --lr_scheduler_type="cosine" --model_path="Open-Orca/Mistral-7B-OpenOrca" --remove_unused_columns=False \
 --sample=30000 --save_strategy "no"
