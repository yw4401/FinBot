import json
import os
from pathlib import Path

import torch
from deepspeed import OnDevice
from huggingface_hub import snapshot_download
from transformers import (
    LlamaTokenizer, LlamaTokenizerFast, AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, AutoConfig,
)

import config


class ListDataset(torch.utils.data.Dataset):

    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]


def restrict_article(text, max_token, tokenizer: LlamaTokenizer):
    tokenized = tokenizer.encode(text, add_special_tokens=False)
    if len(tokenized) > max_token:
        return tokenizer.decode(tokenized[:max_token], skip_special_tokens=False)
    return text


def truncate_summary_example_chat(system, question, body, summary, tokenizer, max_context,
                                  buffer=20, template=None):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": format_llama_sum_user(question, "")},
        {"role": "assistant", "content": format_llama_sum_resp(summary)}
    ]
    body_tokens = max_context - len(tokenizer.apply_chat_template(messages, chat_template=template)) - buffer
    return restrict_article(body, body_tokens, tokenizer)


def truncate_summary_example_plain(question, body, summary, tokenizer, max_context,
                                   input_template=config.PLAIN_INPUT_TEMPLATE,
                                   output_template=config.PLAIN_OUTPUT_TEMPLATE, seq2seq=True, buffer=20):
    final_input = input_template.format(body=body, question=question)
    final_output = output_template.format(summary=summary)
    if seq2seq:
        final_text = final_input
    else:
        final_text = final_input + "\n" + final_output
    body_tokens = max_context - len(tokenizer.encode(final_text, add_special_tokens=False)) - buffer
    return restrict_article(body, body_tokens, tokenizer)


def format_llama_sum_user(question, body):
    return config.LLAMA_USER_TEMPLATE.format(context=body, question=question)


def format_llama_sum_resp(summary):
    return config.LLAMA_S_TEMPLATE.format(summary=summary)


def format_summary_example(example, tokenizer, template=None):
    output_texts = []
    for i in range(len(example['body'])):
        q = example["question"][i]
        c = example["body"][i]
        s = format_llama_sum_resp(example["summary"][i])

        user = format_llama_sum_user(q, c)
        text = tokenizer.apply_chat_template([
            {"role": "system", "content": config.LLAMA_SUMMARY_BULLET_INSTRUCTION},
            {"role": "user", "content": user},
            {"role": "assistant", "content": s}
        ], tokenize=False, chat_template=template)
        if text[:len(tokenizer.bos_token)] == tokenizer.bos_token:
            text = text[len(tokenizer.bos_token):]
        if text[-len(tokenizer.eos_token):] == tokenizer.eos_token:
            text = text[:-len(tokenizer.eos_token)]
        output_texts.append(text)

    return output_texts


def find_target_modules(model):
    # Initialize a Set to Store Unique Layers
    unique_layers = set()

    # Iterate Over All Named Modules in the Model
    for name, module in model.named_modules():
        # Check if the Module Type Contains 'Linear4bit'
        if "Linear4bit" in str(type(module)):
            # Extract the Type of the Layer
            layer_type = name.split('.')[-1]

            # Add the Layer Type to the Set of Unique Layers
            unique_layers.add(layer_type)

    # Return the Set of Unique Layers Converted to a List
    return list(unique_layers)


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=()):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False


class DSPipeline:
    '''
    Example helper class for comprehending DeepSpeed Meta Tensors, meant to mimic HF pipelines.
    The DSPipeline can run with and without meta tensors.
    '''

    def __init__(self,
                 model_name='bigscience/bloom-3b',
                 dtype=torch.float16,
                 is_meta=True,
                 device=-1,
                 checkpoint_path=None,
                 trust_remote_code=False,
                 token=None,
                 model_type=AutoModelForCausalLM
                 ):
        self.model_name = model_name
        self.dtype = dtype
        self.token = token

        if isinstance(device, torch.device):
            self.device = device
        elif isinstance(device, str):
            self.device = torch.device(device)
        elif device < 0:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{device}")

        # the Deepspeed team made these so it's super fast to load (~1 minute), rather than wait 10-20min loading time.
        self.tp_presharded_models = ["microsoft/bloom-deepspeed-inference-int8",
                                     "microsoft/bloom-deepspeed-inference-fp16"]

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left",
                                                       trust_remote_code=trust_remote_code, token=token)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if is_meta:
            '''When meta tensors enabled, use checkpoints'''
            self.repo_root, self.checkpoints_json = self._generate_json(checkpoint_path)
            self.config = AutoConfig.from_pretrained(self.repo_root)
            with OnDevice(dtype=dtype, device="meta", enabled=True):
                self.model = AutoModelForCausalLM.from_config(self.config, torch_dtype=dtype)
                # self.model = model_type.from_pretrained(self.repo_root, torch_dtype=dtype,
                #                                         use_flash_attention_2=True, low_cpu_mem_usage=True, token=token)
            self.model = self.model.eval()
        else:
            self.model = model_type.from_pretrained(self.repo_root, torch_dtype=dtype,
                                                    trust_remote_code=trust_remote_code, token=token)
            self.model.eval()

    def __call__(self,
                 inputs=("test",), generate_kwargs=None):
        if generate_kwargs is None:
            generate_kwargs = {}
        if isinstance(inputs, str):
            input_list = [inputs]
        else:
            input_list = inputs

        outputs = self.generate_outputs(input_list, generate_kwargs)
        return outputs

    def _generate_json(self, checkpoint_path=None):
        if checkpoint_path is None:
            repo_root = snapshot_download(self.model_name,
                                          allow_patterns=["*"],
                                          cache_dir=os.getenv("TRANSFORMERS_CACHE", None),
                                          ignore_patterns=["*.safetensors"],
                                          local_files_only=False,
                                          revision=None, token=self.token)
        else:
            assert os.path.exists(checkpoint_path), f"Checkpoint path {checkpoint_path} does not exist"
            repo_root = checkpoint_path

        if os.path.isfile(os.path.join(repo_root, "ds_inference_config.json")):
            with open(os.path.join(repo_root, "ds_inference_config.json")) as f:
                data = json.load(f)
            data["base_dir"] = repo_root
            return data
        else:
            checkpoint_files = [
                str(entry).split("/")[-1]
                for entry in Path(repo_root).rglob("*.[bp][it][n]") if entry.is_file()
            ]
            data = {
                "type": "DS_MODEL",
                "checkpoints": checkpoint_files,
                "version": 1.0,
                "base_dir": repo_root,
            }

        return repo_root, data

    def generate_outputs(self, inputs=("test",), generate_kwargs=None):
        if generate_kwargs is None:
            generate_kwargs = {}
        input_tokens = self.tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=True)
        for t in input_tokens:
            if torch.is_tensor(input_tokens[t]):
                input_tokens[t] = input_tokens[t].to(self.device)

        self.model.cuda().to(self.device)

        if isinstance(self.tokenizer, LlamaTokenizerFast):
            # NOTE: Check if Llamma can work w/ **input_tokens
            #       'token_type_ids' kwarg not recognized in Llamma generate function
            outputs = self.model.generate(input_tokens.input_ids, **generate_kwargs)
        else:
            outputs = self.model.generate(**input_tokens, **generate_kwargs)
        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return outputs
