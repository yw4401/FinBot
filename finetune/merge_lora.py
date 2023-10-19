from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType


if __name__ == "__main__":
    model_check = "./flan-t5-xl"
    lora_check = "./t5-xl-finetuned-summary/checkpoint-4276"
    model_out = "t5-summary-xl"
    MAX_BODY_TOKEN = 2048
    
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=32, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q", "v"],
    )
    print("Loading Base Model")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_check)
    print("Loading LORA")
    model = get_peft_model(model, peft_config)
    peft_model = load_state_dict_from_zero_checkpoint(model, lora_check)
    print("Merging Model")
    merged_model = peft_model.merge_and_unload()
    print("Saving Merged Model")
    merged_model.save_pretrained(model_out)
    print("Saving Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_check, model_max_length=MAX_BODY_TOKEN)
    tokenizer.save_pretrained(model_out)
    
