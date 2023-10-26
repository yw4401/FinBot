from dataclasses import dataclass, field
from typing import cast, Optional

import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from deepspeed import OnDevice
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    HfArgumentParser, )
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

import config
from common import format_summary_example, truncate_summary_example_chat


@dataclass
class ScriptArguments:
    model_path: Optional[str] = field(default="meta-llama/Llama-2-7b-chat-hf")
    token_path: Optional[str] = field(default="./hf_token")
    dataset_path: Optional[str] = field(default="./fine-tune-summary-train.parquet")
    sample: Optional[int] = field(default=50000)
    model_max_length: Optional[int] = field(default=2048)
    eval_size: Optional[float] = field(default=1000)
    lora_target: Optional[str] = field(default="q_proj,v_proj")
    lora_r: Optional[int] = field(default=16)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.05)
    cache_dir: Optional[str] = field(default="./transformers")
    start_text: Optional[str] = field(default="<|im_start|> assistant")


def main():
    parser = HfArgumentParser([TrainingArguments, ScriptArguments])
    train_args, script_args = parser.parse_args_into_dataclasses()
    train_args: TrainingArguments = cast(TrainingArguments, train_args)
    script_args: ScriptArguments = cast(ScriptArguments, script_args)
    script_args.lora_target = script_args.lora_target.split(",")

    with open(script_args.token_path, "r") as fp:
        hf_token = fp.read().strip()

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_path, token=hf_token,
                                              model_max_length=script_args.model_max_length,
                                              cache_dir=script_args.cache_dir,
                                              padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token

    # loading and prepare dataset
    train_df = pd.read_parquet("fine-tune-summary-train.parquet").sample(n=script_args.sample, random_state=93)
    train_df["body"] = train_df.apply(
        lambda row: truncate_summary_example_chat(system=config.LLAMA_SUMMARY_BULLET_INSTRUCTION,
                                                  question=row["question"],
                                                  body=row["body"],
                                                  summary=row["summary"],
                                                  tokenizer=tokenizer,
                                                  max_context=script_args.model_max_length), axis=1)
    print(train_df.head())
    print(train_df.summary.iloc[0])
    train_data = Dataset.from_pandas(train_df[["body", "question", "summary"]])
    raw_datasets = DatasetDict({
        "train": train_data
    })

    # preparing lora configuration
    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        bias="none",
        task_type="CASUAL_LM",
        target_modules=script_args.lora_target,
    )

    # loading the base model
    with OnDevice(dtype=torch.bfloat16, device="meta", enabled=True):
        model = AutoModelForCausalLM.from_pretrained(
            script_args.model_path, use_cache=not train_args.gradient_checkpointing, token=hf_token,
            torch_dtype=torch.bfloat16, use_flash_attention_2=True, low_cpu_mem_usage=True,
            cache_dir=script_args.cache_dir)
    if train_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    print(raw_datasets["train"])
    # creating trainer with collator
    collator = DataCollatorForCompletionOnlyLM(script_args.start_text, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model, args=train_args, train_dataset=raw_datasets["train"],
        formatting_func=lambda x: format_summary_example(x, tokenizer),
        data_collator=collator, tokenizer=tokenizer,
        max_seq_length=script_args.model_max_length, peft_config=peft_config, packing=False
    )
    # trainer.accelerator.print(f"{trainer.model}")
    trainer.model.print_trainable_parameters()

    # start training
    trainer.train()

    # save model on main process
    trainer.accelerator.wait_for_everyone()
    if trainer.accelerator.is_main_process:
        trainer.save_model()
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    with open("./hf_token", "r") as fp:
        hf_token = fp.read().strip()

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=hf_token,
                                              model_max_length=2048)
    print(torch.tensor([tokenizer.eos_token_id]))
