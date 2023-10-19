import re

import nltk
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from evaluate import load
import torch
from transformers import AutoTokenizer
from transformers import Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, GenerationConfig
from transformers import Seq2SeqTrainer
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType


def preprocess_function(examples, max_input_length, max_target_length):
    model_inputs = tokenizer(examples["body"], max_length=max_input_length, truncation=True)
    labels = tokenizer(text_target=examples["summary"], max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


def compute_metrics(eval_pred):
    metric = load("rouge")
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    # Note that other metrics may not have a `use_aggregator` parameter
    # and thus will return a list, computing a metric for each sentence.
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True,
                            use_aggregator=True)
    # Extract a few results
    result = {key: value * 100 for key, value in result.items()}

    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


if __name__ == "__main__":
    model_name = "t5-xxl"
    cache_dir = "./transformers"
    model_checkpoint = "../../flan-t5-xxl"
    generation_checkpoint = model_checkpoint
    BATCH_TRAIN = 2
    BATCH_EVAL = 4
    GRADIENT_CHECKPOINT = False
    GRADIENT_STEP = 1
    LEARNING_RATE = 1e-3
    EPOCHS = 4
    WARM_UP = 200
    MAX_BODY_TOKEN = 2048
    MAX_SUMMARY_TOKEN = 256
    TEMPERATURE = 0

    config = GenerationConfig.from_pretrained(generation_checkpoint, cache_dir=cache_dir)
    config.max_new_tokens = MAX_SUMMARY_TOKEN
    args = Seq2SeqTrainingArguments(
        f"{model_name}-finetuned-summary",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_TRAIN,
        per_device_eval_batch_size=BATCH_EVAL,
        gradient_accumulation_steps=GRADIENT_STEP,
        gradient_checkpointing=GRADIENT_CHECKPOINT,
        num_train_epochs=EPOCHS,
        optim="adafactor",
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=WARM_UP,
        predict_with_generate=True,
        save_strategy="epoch",
        deepspeed="deepsp.json",
        bf16=True,
        generation_config=config,
        seed=93
    )
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=32, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q", "v"],
    )
    
    train_df = pd.read_parquet("fine-tune-summary-train.parquet")
    print(train_df.head())
    print(train_df.summary.iloc[0])
    
    train_data = Dataset.from_pandas(train_df[["body", "summary"]])
    raw_datasets = DatasetDict({
        "train": train_data
    })

    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, cache_dir=cache_dir).half()
    model.generation_config = config
    if generation_checkpoint:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model = get_peft_model(model, peft_config)
    
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, model_max_length=MAX_BODY_TOKEN, cache_dir=cache_dir)
    max_input_length = tokenizer.model_max_length
    max_target_length = MAX_SUMMARY_TOKEN
    tokenized_datasets = raw_datasets.map(lambda r: preprocess_function(r, max_input_length, max_target_length),
                                          batched=True)
    print(f"Truncating to {max_input_length}")
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    model.print_trainable_parameters()

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    try:
        results = trainer.train(resume_from_checkpoint=True)
    except (ValueError, FileNotFoundError) as e:
        results = trainer.train(resume_from_checkpoint=False)
        
    trainer.save_model()




