import re

import nltk
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from evaluate import load
from transformers import AutoTokenizer
from transformers import Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType


def preprocess_function(examples, max_input_length, max_target_length):
    inputs = []
    for body, type in zip(examples["body"], examples["summary_type"]):
        if type == "BULLETS":
            inputs.append(prefix_bullets + body)
        elif type == "PLAIN":
            inputs.append(prefix_plain + body)
        else:
            raise ValueError("typo")

    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
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
    sample_df = pd.read_parquet("gs://scraped-news-article-data-null/fine-tune-summary--1.parquet")
    sample_df = sample_df.sample(frac=1, random_state=93).reset_index(drop=True)
    clean_regex = re.compile(r"\*[\s\n]*(?=\*)")
    sample_df["summary"] = sample_df.summary.apply(lambda s: clean_regex.sub(" ", s).strip())
    train_df = sample_df.iloc[:21125]
    eval_df = sample_df.iloc[21125:]

    model_checkpoint = "google/flan-t5-base"
    train_data = Dataset.from_pandas(train_df[["body", "summary", "summary_type"]])
    eval_data = Dataset.from_pandas(eval_df[["body", "summary", "summary_type"]])
    raw_datasets = DatasetDict({
        "train": train_data,
        "eval": eval_data
    })

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    prefix_bullets = "summarize in bullet points: "
    prefix_plain = "summarize as paragraph: "
    max_input_length = 2048
    max_target_length = 512
    tokenized_datasets = raw_datasets.map(lambda r: preprocess_function(r, max_input_length, max_target_length),
                                          batched=True)

    model_name = "t5"
    BATCH_TRAIN = 2
    BATCH_EVAL = 4
    GRADIENT_STEP = 1
    LEARNING_RATE = 2e-5
    EPOCHS = 4
    LAMBDA = 0.01

    args = Seq2SeqTrainingArguments(
        f"{model_name}-finetuned-summary",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_TRAIN,
        per_device_eval_batch_size=BATCH_EVAL,
        weight_decay=LAMBDA,
        num_train_epochs=EPOCHS,
        predict_with_generate=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        bf16=True,
        seed=93
    )
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=16, lora_alpha=16, lora_dropout=0.05, bias="none",
        target_modules=["q", "v"],
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    model = get_peft_model(model, peft_config)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    model.print_trainable_parameters()

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["eval"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    try:
        results = trainer.train(resume_from_checkpoint=True)
    except ValueError as e:
        results = trainer.train(resume_from_checkpoint=False)

    trainer.save_model()




