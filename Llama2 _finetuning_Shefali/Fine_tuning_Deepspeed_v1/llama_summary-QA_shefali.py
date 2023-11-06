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
from common import format_summary_example, truncate_summary_example_chat, format_qa_example
from datasets import load_dataset, load_metric, DatasetDict

@dataclass
class ScriptArguments:
    model_path: Optional[str] = field(default="meta-llama/Llama-2-7b-chat-hf")
    token_path: Optional[str] = field(default="./hf_token")
    #dataset_path: Optional[str] = field(default="./fine-tune-summary-train.parquet")
    sample: Optional[int] = field(default=50000)
    model_max_length: Optional[int] = field(default=4096)
    lora_target: Optional[str] = field(default="q_proj,v_proj")
    lora_r: Optional[int] = field(default=16)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.05)
    cache_dir: Optional[str] = field(default="./transformers")
    buffer_len: Optional[str] = field(default=20)
    start_text: Optional[str] = field(default="[/INST]")


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
                                              add_eos_token=True,
                                              cache_dir=script_args.cache_dir,
                                              padding_side="right")
    tokenizer.pad_token = "[PAD]"

    # loading and prepare dataset
    #train = load_dataset('squad_v2', split='train')
    #train_df = pd.DataFrame(train)
    #prompt_text1 = "Based on the following context. Context_start: "
    #prompt_text2 = " Context_end.\nAnswer the following question Question_start "
    #prompt_text3 = " Question_end.\nThe answer to this question is: "
    #train_df['answers_text'] = train_df['answers'].apply(lambda x: ' '.join(x['text']))
    #if train_df.shape[0] > script_args.sample:
        #train_df = train_df.sample(n=script_args.sample, random_state=93)
    #train_df["body"] = train_df.apply(
        #lambda row: truncate_summary_example_chat(system=config.LLAMA_SUMMARY_BULLET_INSTRUCTION,
                                                  #question=row["question"],
                                                  #body=row["body"],
                                                  #summary=row["summary"],
                                                  #tokenizer=tokenizer,
                                                  #max_context=script_args.model_max_length,
                                                  #buffer=script_args.buffer_len), axis=1)
    #print(train_df.head())
    #print(train_df.prompt.iloc[0])
    #train_data = Dataset.from_pandas(train_df[["context", "question", "answers_text", "prompt"]])
    #raw_datasets = DatasetDict({
        #"train": train_data
    #})

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
    with OnDevice(dtype=torch.float16, device="meta", enabled=True):
        model = AutoModelForCausalLM.from_pretrained(
            script_args.model_path, use_cache=not train_args.gradient_checkpointing, token=hf_token,
            torch_dtype=torch.float16, use_flash_attention_2=False,
            cache_dir=script_args.cache_dir)
    if train_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    #print(raw_datasets["train"])
    # creating trainer with collator
    collator = DataCollatorForCompletionOnlyLM(script_args.start_text, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model, args=train_args, train_dataset=load_dataset('squad_v2', split='train[:100]'),
        formatting_func=lambda x: format_qa_example(x, tokenizer),
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
