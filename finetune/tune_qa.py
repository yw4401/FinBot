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
    HfArgumentParser,
    GenerationConfig
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

import config
from common import truncate_qa_example_chat, format_qa_example


@dataclass
class ScriptArguments:
    model_path: Optional[str] = field(default="meta-llama/Llama-2-7b-chat-hf")
    token_path: Optional[str] = field(default="./hf_token")
    dataset_path: Optional[str] = field(default="./fine-tune-qa-train.parquet")
    template_path: Optional[str] = field(default=None)
    flash_attention: Optional[bool] = field(default=True)
    use_meta: Optional[bool] = field(default=True)
    sample: Optional[int] = field(default=50000)
    model_max_length: Optional[int] = field(default=4096 - 256)
    model_max_new_tokens: Optional[int] = field(default=256)
    lora_target: Optional[str] = field(default="q_proj,k_proj,v_proj,o_proj")
    lora_r: Optional[int] = field(default=16)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.05)
    cache_dir: Optional[str] = field(default="./transformers")
    buffer_len: Optional[str] = field(default=20)
    start_text: Optional[str] = field(default="[/INST] ")


def main():
    parser = HfArgumentParser([TrainingArguments, ScriptArguments])
    train_args, script_args = parser.parse_args_into_dataclasses()
    train_args: TrainingArguments = cast(TrainingArguments, train_args)
    script_args: ScriptArguments = cast(ScriptArguments, script_args)
    script_args.lora_target = script_args.lora_target.split(",")

    with open(script_args.token_path, "r") as fp:
        hf_token = fp.read().strip()

    if script_args.template_path:
        with open(script_args.template_path, "r") as fp:
            template = fp.read()
    else:
        template = None

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_path, token=hf_token,
                                              model_max_length=script_args.model_max_length,
                                              cache_dir=script_args.cache_dir,
                                              padding_side="right")
    tokenizer.pad_token = "[PAD]"

    # loading and prepare dataset
    data_df = pd.read_parquet(script_args.dataset_path)
    if data_df.shape[0] > script_args.sample:
        data_df = data_df.sample(n=script_args.sample, random_state=93)
    data_df["context"] = data_df.apply(
        lambda row: truncate_qa_example_chat(system=config.LLAMA_SUMMARY_BULLET_INSTRUCTION,
                                             question=row["question"],
                                             context=row["context"],
                                             answer=row["answer"],
                                             tokenizer=tokenizer,
                                             template=template,
                                             max_context=script_args.model_max_length,
                                             buffer=script_args.buffer_len), axis=1)
    print(data_df.head())
    print(data_df.context.iloc[0])
    print(data_df.question.iloc[0])
    print(data_df.answer.iloc[0])
    # train_df = data_df
    train_data = Dataset.from_pandas(data_df[["context", "question", "answer"]])
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
    if script_args.use_meta:
        with OnDevice(dtype=torch.bfloat16, device="meta", enabled=True):
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    script_args.model_path, use_cache=not train_args.gradient_checkpointing, token=hf_token,
                    torch_dtype=torch.bfloat16, use_flash_attention_2=script_args.flash_attention,
                    low_cpu_mem_usage=True,
                    cache_dir=script_args.cache_dir)
            except ValueError:
                model = AutoModelForCausalLM.from_pretrained(
                    script_args.model_path, use_cache=not train_args.gradient_checkpointing, token=hf_token,
                    torch_dtype=torch.bfloat16, use_flash_attention_2=script_args.flash_attention,
                    cache_dir=script_args.cache_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            script_args.model_path, use_cache=not train_args.gradient_checkpointing, token=hf_token,
            torch_dtype=torch.bfloat16, use_flash_attention_2=script_args.flash_attention,
            cache_dir=script_args.cache_dir)
    if train_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    print(model)
    print(raw_datasets["train"])
    # creating trainer with collator
    collator = DataCollatorForCompletionOnlyLM(script_args.start_text, tokenizer=tokenizer)
    try:
        gen_config = GenerationConfig.from_pretrained(script_args.model_path, cache_dir=script_args.cache_dir)
    except OSError:
        gen_config = GenerationConfig.from_model_config(model.config)
    gen_config.max_new_tokens = script_args.model_max_new_tokens
    train_args.generation_config = gen_config
    model.generation_config = gen_config
    trainer = SFTTrainer(
        model=model, args=train_args, train_dataset=raw_datasets["train"],
        formatting_func=lambda x: format_qa_example(x, tokenizer, template),
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


def test_tokenizer(model):
    with open("./hf_token", "r") as fp:
        hf_token = fp.read().strip()

    tokenizer = AutoTokenizer.from_pretrained("Open-Orca/Mistral-7B-OpenOrca", token=hf_token,
                                              model_max_length=2048, add_eos_token=True, padding=True)
    tokenizer.pad_token = tokenizer.eos_token
    message_example = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Who are you"},
        {"role": "assistant", "content": "A helpful assistant"}
    ]
    chat_applied = tokenizer.apply_chat_template(message_example, tokenize=False)
    text = chat_applied
    if text[:len(tokenizer.bos_token)] == tokenizer.bos_token:
        text = text[len(tokenizer.bos_token):]
    if text[-len(tokenizer.eos_token):] == tokenizer.eos_token:
        text = text[:-len(tokenizer.eos_token)]
    print(tokenizer("test"))
    print(chat_applied)
    print(tokenizer(text))
    for id in tokenizer(text)["input_ids"]:
        print(tokenizer.decode([id]), end="")


if __name__ == "__main__":
    main()
