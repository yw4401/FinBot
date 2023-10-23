from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
import torch


if __name__ == "__main__":
    model_check = "meta-llama/Llama-2-7b-chat-hf"
    lora_check = "./summary-llama-lora/checkpoint-2500"
    model_out = "./summary-llama"
    
    peft_config = LoraConfig
    print("Loading Base Model")
    model = AutoModelForCausalLM.from_pretrained(model_check)
    print("Loading LORA")
    peft_config = PeftConfig.from_pretrained(lora_check)
    model = get_peft_model(model, peft_config)
    peft_model = load_state_dict_from_zero_checkpoint(model, lora_check)
    print("Merging Model")
    merged_model = peft_model.merge_and_unload()
    print("Saving Merged Model")
    merged_model.save_pretrained(model_out)
    print("Saving Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_check)
    tokenizer.save_pretrained(model_out)
    
