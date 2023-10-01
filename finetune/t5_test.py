import re

import nltk
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from evaluate import load
from transformers import AutoTokenizer
from transformers import Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, AutoConfig
from transformers import Seq2SeqTrainer
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
from t5_summary import get_data_sets_df
from transformers import pipeline
import deepspeed
import os
import torch
from tqdm import tqdm


prefix_bullets = "summarize in bullet points: "
prefix_plain = "summarize as paragraph: "


def prepend_command(row):
    body = row["body"]
    if row["summary_type"] == "BULLETS":
        body = prefix_bullets + body
    elif row["summary_type"] == "PLAIN":
        body = prefix_plain + body
    else:
        raise ValueError("typo")
        
    return body


class ListDataset(torch.utils.data.Dataset):
    
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]


if __name__ == "__main__":
    _, _, test_df = get_data_sets_df("gs://scraped-news-article-data-null/fine-tune-summary--1.parquet")
    model_checkpoint = "./t5-summary"
    out_file = "test_predicted.parquet"
    batch = 4

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    max_input_length = tokenizer.model_max_length
    max_target_length = 256
    print(f"Truncating to {max_input_length}")
    test_df["body"] = test_df.apply(prepend_command, axis=1)
    test_body = ListDataset(test_df["body"].tolist())
    print(test_df["body"].head())
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, torch_dtype=torch.half, device_map="auto")
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    results = []
    newline_regex = re.compile(r"\s*\*")
    counter = 0
    
    for out in tqdm(summarizer(test_body, batch_size=batch, truncation=True), total=len(test_body)):
        out = [newline_regex.sub("\n*", o["summary_text"]) for o in out]
        if counter % 100 == 0:
            for o in out:
                print(o)
        results.extend(out)
        counter += 1
    test_df["predicted"] = results
    test_df.to_parquet(out_file, index=False)
    
    