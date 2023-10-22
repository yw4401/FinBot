from dataclasses import dataclass, field
from typing import cast, Optional, List
import torch
import config
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    HfArgumentParser,
)
from trl import SFTTrainer
import transformers
from peft import LoraConfig, get_peft_model
import pandas as pd


@dataclass
class ScriptArguments:
    model_path: Optional[str] = field(default="./Llama-2-7b-chat-hf")
    token_path: Optional[str] = field(default="./hf_token")
    dataset_path: Optional[str] = field(default="./fine-tune-summary-train.parquet")
    sample: Optional[int] = field(default=50000)
    eval_size: Optional[float] = field(default=1000)
    lora_target: Optional[List[str]] = ("q_proj", "v_proj")
    cache_dir: Optional[str] = "./transformers"
    lora_r: Optional[int] = field(default=32)
    lora_alpha: Optional[int] = field(default=32)
    lora_dropout: Optional[float] = field(default=0.10)


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."


def format_prompt(examples):
    if examples[0]["role"] == "system":
        examples = [
                       {
                           "role": examples[1]["role"],
                           "content": B_SYS
                                      + examples[0]["content"]
                                      + E_SYS
                                      + examples[1]["content"],
                       }
                   ] + examples[2:]
    assert all([msg["role"] == "user" for msg in examples[::2]]) and all(
        [msg["role"] == "assistant" for msg in examples[1::2]]
    ), (
        "model only supports 'system', 'user' and 'assistant' roles, "
        "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
    )
    dialog_texts: List[str] = [
            f"<s>{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} </s>"
            for prompt, answer in zip(
                examples[::2],
                examples[1::2],
            )
    ]
    return "".join(dialog_texts)


def format_example(example):
    q_header = "### Question"
    c_header = "### Context"

    q = example["question"]
    c = example["body"]

    user = f"{q_header}\n{q}\n\n{c_header}\n{c}"

    example["text"] = format_prompt([
        {"role": "system", "content": config.LLAMA_SUMMARY_BULLET_INSTRUCTION},
        {"role": "user", "content": user},
        {"role": "assistant", "content": example["summary"]}
    ])
    return example


def prepare_dataset(dataset, formatter_func, tokenizer):
    # formatting each sample
    dataset_prepared = dataset.map(
        formatter_func
    )

    return dataset_prepared


def main():
    parser = HfArgumentParser([TrainingArguments, ScriptArguments])
    train_args, script_args = parser.parse_args_into_dataclasses()
    train_args: TrainingArguments = cast(TrainingArguments, train_args)
    script_args: ScriptArguments = cast(ScriptArguments, script_args)

    with open(script_args.token_path, "r") as fp:
        hf_token = fp.read().strip()

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_path, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # loading and prepare dataset
    train_df = pd.read_parquet("fine-tune-summary-train.parquet").sample(n=script_args.sample, random_state=93)
    print(train_df.head())
    print(train_df.summary.iloc[0])
    train_data = Dataset.from_pandas(train_df[["body", "question", "summary"]])
    dataset = prepare_dataset(
        dataset=train_data,
        formatter_func=format_example,
        tokenizer=tokenizer,
    )
    raw_datasets = DatasetDict({
        "train": dataset
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
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_path, use_cache=not train_args.gradient_checkpointing, token=hf_token, torch_dtype=torch.bfloat16
    )
    if train_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    print(raw_datasets["train"])
    # creating trainer
    trainer = SFTTrainer(
        model=model, args=train_args, train_dataset=raw_datasets["train"], dataset_text_field="text", max_seq_length=4096, peft_config=peft_config,
    )
    # trainer.accelerator.print(f"{trainer.model}")
    trainer.model.print_trainable_parameters()

    # start training
    trainer.train()

    # save model on main process
    trainer.accelerator.wait_for_everyone()
    # save everything else on main process
    if trainer.args.process_index == 0:
        trainer.save_model()


if __name__ == "__main__":
    main()
