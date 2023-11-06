import torch
from dataclasses import dataclass, field
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from peft import PeftModel, PeftConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import (
    HfArgumentParser, )
from typing import Optional, cast


@dataclass
class ScriptArguments:
    model_path: Optional[str] = field(default="meta-llama/Llama-2-7b-chat-hf")
    token_path: Optional[str] = field(default="./hf_token")
    lora_path: Optional[str] = field(default="./summary-llama-lora")
    cache_dir: Optional[str] = field(default="./transformers")
    model_out: Optional[str] = field(default="./summary-llama")
    deepspeed: Optional[bool] = field(default=False)


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
    if script_args.deepspeed:
        config = PeftConfig.from_pretrained(script_args.lora_path)
        peft_model = get_peft_model(model, config)
        state_dict = get_fp32_state_dict_from_zero_checkpoint(script_args.lora_path)  # already on cpu
        peft_model.load_state_dict(state_dict)
    else:
        peft_model = PeftModel.from_pretrained(model, script_args.lora_path)
    print("Merging Model")
    merged_model = peft_model.merge_and_unload()
    merged_model = merged_model.to(dtype=torch.float16)
    print("Saving Merged Model")
    merged_model.save_pretrained(script_args.model_out)
    print("Saving Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_path)
    tokenizer.save_pretrained(script_args.model_out)
