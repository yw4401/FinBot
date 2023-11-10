#!/bin/bash
gsutil cp gs://scraped-news-article-data-null/fine-tune-summary-train.parquet .
gsutil cp gs://scraped-news-article-data-null/fine-tune-summary-test.parquet .
python tune_summary.py --output_dir="./summary-dolly-lora" \
 --per_device_train_batch_size=1 --learning_rate=1e-4 --num_train_epochs=4 --weight_decay=0.1 --warmup_steps=200 \
 --bf16=True --lr_scheduler_type="cosine" --model_path="databricks/dolly-v2-3b" --remove_unused_columns=False \
 --sample=100 --validation 10 --save_strategy "epoch" --lora_target "query_key_value" --lora_r 16 --lora_alpha 16 \
 --gradient_checkpointing True --start_text "### Response:" --template ./dolly.template --flash_attention False \
 --use_meta False --model_max_length 2048
