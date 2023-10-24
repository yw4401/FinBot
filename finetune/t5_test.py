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
from transformers import pipeline
import deepspeed
import os
import torch
from tqdm import tqdm
from transformers.models.t5.modeling_t5 import T5Block

from finetune.common import ListDataset

if __name__ == "__main__":
    test_df = pd.read_parquet("fine-tune-summary-test.parquet")
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    model_checkpoint = "./t5-summary-xl"
    model_name = "summary-t5-xl"
    out_file = f"{model_name}-test-predicted.parquet"
    batch = 12
    MAX_BODY_TOKEN = 2048
    MAX_SUMMARY_TOKEN = 256
    DTYPE = torch.float16
    TEMPERATURE = 0

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, model_max_length=MAX_BODY_TOKEN)
    max_input_length = tokenizer.model_max_length
    max_target_length = MAX_SUMMARY_TOKEN
    print(f"Truncating to {max_input_length}")
    test_body = ListDataset(test_df["body"].tolist())
    print(test_df["body"].head())
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, torch_dtype=DTYPE)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, temperature=TEMPERATURE,
                          max_new_tokens=MAX_SUMMARY_TOKEN, device=local_rank)
    summarizer.model = deepspeed.init_inference(
        summarizer.model,
        mp_size=world_size,
        dtype=DTYPE,
        max_tokens=MAX_BODY_TOKEN,
        injection_policy={T5Block: ('SelfAttention.o', 'EncDecAttention.o', 'DenseReluDense.wo')}
    )

    results = []
    newline_regex = re.compile(r"\s*\*")
    counter = 0
    
    for out in tqdm(summarizer(test_body, batch_size=batch, truncation=True), total=len(test_body)):
        out = [newline_regex.sub("\n*", o["summary_text"]).strip() for o in out]
        if counter % 100 == 0:
            for o in out:
                print(o)
        results.extend(out)
        counter += 1
    test_df["predicted"] = results
    test_df.to_parquet(out_file, index=False)
    
    
