from dataclasses import dataclass, field
from typing import cast, Optional, List

import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    HfArgumentParser,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

import config


@dataclass
class ScriptArguments:
    model_path: Optional[str] = field(default="./Llama-2-7b-chat-hf")
    token_path: Optional[str] = field(default="./hf_token")
    dataset_path: Optional[str] = field(default="./fine-tune-summary-train.parquet")
    sample: Optional[int] = field(default=50000)
    model_max_length: Optional[int] = field(default=2048)
    eval_size: Optional[float] = field(default=1000)
    lora_target: Optional[List[str]] = field(default=("q_proj", "k_proj", "v_proj"))
    cache_dir: Optional[str] = field(default="./transformers")
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
    s_header = "### Summary\n"

    output_texts = []
    for i in range(len(example['body'])):
        q = example["question"][i]
        c = example["body"][i]
        s = s_header + example["summary"][i]

        user = f"{q_header}\n{q}\n\n{c_header}\n{c}"
        text = format_prompt([
            {"role": "system", "content": config.LLAMA_SUMMARY_BULLET_INSTRUCTION},
            {"role": "user", "content": user},
            {"role": "assistant", "content": s}
        ])
        output_texts.append(text)

    return output_texts


def main():
    parser = HfArgumentParser([TrainingArguments, ScriptArguments])
    train_args, script_args = parser.parse_args_into_dataclasses()
    train_args: TrainingArguments = cast(TrainingArguments, train_args)
    script_args: ScriptArguments = cast(ScriptArguments, script_args)

    with open(script_args.token_path, "r") as fp:
        hf_token = fp.read().strip()

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_path, token=hf_token, model_max_length=script_args.model_max_length)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # loading and prepare dataset
    train_df = pd.read_parquet("fine-tune-summary-train.parquet").sample(n=script_args.sample, random_state=93)
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
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_path, use_cache=not train_args.gradient_checkpointing, token=hf_token,
        torch_dtype=torch.bfloat16, use_flash_attention_2=True)
    if train_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    print(raw_datasets["train"])
    # creating trainer with collator
    response_template = "### Summary"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model, args=train_args, train_dataset=raw_datasets["train"], formatting_func=format_example,
        data_collator=collator,
        max_seq_length=script_args.model_max_length, peft_config=peft_config, packing=False
    )
    # trainer.accelerator.print(f"{trainer.model}")
    trainer.model.print_trainable_parameters()

    # start training
    trainer.train()

    # save model on main process
    trainer.accelerator.wait_for_everyone()
    state_dict = trainer.accelerator.get_state_dict(trainer.deepspeed)
    unwrapped_model = trainer.accelerator.unwrap_model(trainer.deepspeed)
    if trainer.accelerator.is_main_process:
        unwrapped_model.save_pretrained(train_args.output_dir, state_dict=state_dict)
    trainer.accelerator.wait_for_everyone()

    # save everything else on main process
    if trainer.args.process_index == 0:
        trainer.model.save_pretrained(train_args.output_dir, safe_serialization=True)
        tokenizer.save_pretrained(train_args.output_dir)


if __name__ == "__main__":
    main()
