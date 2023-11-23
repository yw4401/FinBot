#!/bin/bash
gsutil cp gs://scraped-news-article-data-null/fine-tune-qa-train.parquet .
deepspeed tune_qa.py --deepspeed=deepspeed_llama2.json --output_dir="./summary-mistral-lora" \
 --per_device_train_batch_size=1 --learning_rate=1e-4 --num_train_epochs=2 --weight_decay=0.1 --warmup_steps=200 \
 --bf16=True --lr_scheduler_type="cosine" --remove_unused_columns=False \
 --sample=1000000 --save_strategy "epoch" --gradient_checkpointing True
