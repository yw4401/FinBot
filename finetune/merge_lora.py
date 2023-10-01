from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType


if __name__ == "__main__":
    model_check = "google/flan-t5-xl"
    lora_check = "./t5-finetuned-summary"
    model_out = "t5-summary"
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=8, lora_dropout=0.05, bias="none",
        target_modules=["q", "v"],
    )
    print("Loading Base Model")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_check)
    print("Loading LORA")
    model = get_peft_model(model, peft_config)
    peft_model = load_state_dict_from_zero_checkpoint(model, lora_check).to("cuda")
    print("Merging Model")
    merged_model = peft_model.merge_and_unload()
    print("Saving Merged Model")
    merged_model.save_pretrained(model_out)
    print("Saving Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_check, model_max_length=1024)
    tokenizer.save_pretrained(model_out)
    