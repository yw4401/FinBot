from dataclasses import dataclass, field
from typing import Optional, cast

import torch
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import (
    HfArgumentParser, )


@dataclass
class ScriptArguments:
    model_path: Optional[str] = field(default="meta-llama/Llama-2-7b-chat-hf")
    token_path: Optional[str] = field(default="./hf_token")
    lora_path: Optional[str] = field(default="./summary-llama-lora")
    cache_dir: Optional[str] = field(default="./transformers")
    model_out: Optional[str] = field(default="./summary-llama")


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    script_args: ScriptArguments = cast(ScriptArguments, script_args)

    with open(script_args.token_path, "r") as fp:
        hf_token = fp.read().strip()

    print("Loading Base Model")
    model = AutoModelForCausalLM.from_pretrained(script_args.model_path, torch_dtype=torch.float16,
                                                 low_cpu_mem_usage=True, token=hf_token,
                                                 use_flash_attention_2=True,
                                                 cache_dir=script_args.cache_dir)
    print("Loading LORA")
    peft_model = PeftModel.from_pretrained(model, script_args.lora_path)
    print("Merging Model")
    merged_model = peft_model.merge_and_unload()
    print("Saving Merged Model")
    merged_model.save_pretrained(script_args.model_out)
    print("Saving Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_path)
    tokenizer.save_pretrained(script_args.model_out)
