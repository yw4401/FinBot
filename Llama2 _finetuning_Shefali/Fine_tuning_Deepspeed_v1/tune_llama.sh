#!/bin/bash
gsutil cp gs://scraped-news-article-data-null/fine-tune-summary-train.parquet .
gsutil cp gs://scraped-news-article-data-null/fine-tune-summary-test.parquet .
deepspeed llama_summary-QA_shefali.py --deepspeed=deepspeed_llama2.json --output_dir="./results" \
 --per_device_train_batch_size=1 --learning_rate=1e-4 --num_train_epochs=1 --weight_decay=0.1 --warmup_steps=200 \
 --fp16=True --lr_scheduler_type="cosine" --model_path="meta-llama/Llama-2-7b-chat-hf" --remove_unused_columns=False \
 --sample=30000 --save_strategy "no"
