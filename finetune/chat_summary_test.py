import os
from dataclasses import dataclass, field
from typing import cast, Optional

import deepspeed
import pandas as pd
import torch
import transformers
from tqdm import tqdm
from transformers import (
    HfArgumentParser, StoppingCriteriaList,
)

import config
from common import truncate_summary_example_chat, format_llama_sum_user, DSPipeline, StoppingCriteriaSub


@dataclass
class ScriptArguments:
    model_path: Optional[str] = field(default="meta-llama/Llama-2-7b-hf")
    model_type: Optional[str] = field(default="AutoModelForCausalLM")
    strip_input: Optional[bool] = field(default=True)
    template_path: Optional[str] = field(default=None)
    output_name: Optional[str] = field(default="llama-sum")
    check_path: Optional[str] = field(default=None)
    dataset_path: Optional[str] = field(default="./fine-tune-summary-test.parquet")
    token_path: Optional[str] = field(default="./hf_token")
    dtype: Optional[str] = field(default="float16")
    world_size: Optional[int] = field(default=1)
    kernel_inject: Optional[bool] = field(default=True)
    local_rank: Optional[int] = field(default=0)
    max_input_length: Optional[int] = field(default=2048 - 256)
    max_new_tokens: Optional[int] = field(default=256)
    temperature: Optional[float] = field(default=0)
    stop_criteria: Optional[str] = field(default=None)


def compute_conversation(row, pipeline, chat_template=None, stopping_criteria=None, strip_input=True, new_tokens=256):
    conversation = [{"role": "system", "content": config.LLAMA_SUMMARY_BULLET_INSTRUCTION},
                    {"role": "user", "content": format_llama_sum_user(question=row["question"],
                                                                      body=row["body"])}]
    chat_form = pipeline.tokenizer.apply_chat_template(conversation, chat_template=chat_template,
                                                       tokenize=False, add_generation_prompt=True)
    conversation = pipe(chat_form, dict(max_new_tokens=new_tokens, stopping_criteria=stopping_criteria))
    if strip_input:
        prompt_length = len(
            pipe.tokenizer.decode(
                pipe.tokenizer.encode(chat_form),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        )
    else:
        prompt_length = 0
    return conversation[0][prompt_length:]


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    script_args: ScriptArguments = cast(ScriptArguments, script_args)
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    dtype = getattr(torch, script_args.dtype)
    model_type = getattr(transformers, script_args.model_type)
    template = None
    if script_args.template_path:
        with open(script_args.template_path, "r") as fp:
            template = fp.read()

    test_df = pd.read_parquet(script_args.dataset_path)
    with open(script_args.token_path, "r") as fp:
        hf_token = fp.read().strip()

    pipe = DSPipeline(model_name=script_args.model_path,
                      dtype=dtype,
                      is_meta=True,
                      token=hf_token,
                      device=script_args.local_rank,
                      checkpoint_path=script_args.check_path,
                      model_type=model_type,
                      trust_remote_code=True)
    ds_kwargs = dict(base_dir=pipe.repo_root, checkpoint=pipe.checkpoints_json)
    pipe.model = deepspeed.init_inference(pipe.model,
                                          dtype=dtype,
                                          replace_with_kernel_inject=script_args.kernel_inject,
                                          max_tokens=script_args.max_input_length + script_args.max_new_tokens,
                                          tensor_parallel={
                                              "tp_size": int(world_size)
                                          },
                                          **ds_kwargs
                                          )
    if script_args.stop_criteria:
        stop_words = script_args.stop_criteria.split(",")
        stop_words_ids = [pipe.tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze() for stop_word in
                          stop_words]
        stop_words_ids = [torch.reshape(s, (1,)) if s.dim() == 0 else s for s in stop_words_ids]
        print(stop_words_ids)
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    else:
        stopping_criteria = StoppingCriteriaList(
            [StoppingCriteriaSub(stops=[torch.tensor([pipe.tokenizer.eos_token_id])])])

    test_df["body"] = test_df.apply(
        lambda row: truncate_summary_example_chat(system=config.LLAMA_SUMMARY_BULLET_INSTRUCTION,
                                                  question=row["question"],
                                                  body=row["body"],
                                                  summary=row["summary"],
                                                  tokenizer=pipe.tokenizer,
                                                  max_context=script_args.max_input_length), axis=1)

    results = []
    for i, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
        raw_out: str = compute_conversation(row, pipe, chat_template=template, stopping_criteria=stopping_criteria,
                                            new_tokens=script_args.max_new_tokens)
        results.append(raw_out)

        if i % 10 == 0 and local_rank == 0:
            print(raw_out)

    test_df["predicted"] = results
    out_file = f"{script_args.output_name}-test-predicted.parquet"
    test_df.to_parquet(out_file, index=False)
