import os
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import cast, Optional

import deepspeed
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    HfArgumentParser, AutoModelForCausalLM, pipeline, ConversationalPipeline, Conversation,
)

from common import truncate_summary_example, format_llama_sum_user
from finetune import config


@dataclass
class ScriptArguments:
    model_path: Optional[str] = field(default="meta-llama/Llama-2-7b-hf")
    output_name: Optional[str] = field(default="llama-sum")
    dataset_path: Optional[str] = field(default="./fine-tune-summary-test.parquet")
    token_path: Optional[str] = field(default="./hf_token")
    model_max_length: Optional[int] = field(default=2048)
    max_new_tokens: Optional[int] = field(default=256 + 3)
    temperature: Optional[float] = field(default=0)


def compute_conversation(row, pipeline):
    conversation = Conversation([{"role": "system", "content": config.LLAMA_SUMMARY_BULLET_INSTRUCTION},
                                 {"role": "user", "content": format_llama_sum_user(question=row["question"],
                                                                                   body=row["body"])}])
    conversation = pipeline(conversation)
    return conversation.generated_responses[-1]


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    script_args: ScriptArguments = cast(ScriptArguments, script_args)
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    world_size = int(os.getenv('WORLD_SIZE', '1'))

    test_df = pd.read_parquet(script_args.model_path)
    with open(script_args.token_path, "r") as fp:
        hf_token = fp.read().strip()

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_path, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(script_args.model_path, torch_dtype=torch.float16)

    test_df["body"] = test_df.apply(lambda row: truncate_summary_example(row["question"],
                                                                         row["body"],
                                                                         row["summary"], tokenizer,
                                                                         script_args.model_max_length), axis=1)
    summarizer = pipeline("conversational", model=model, tokenizer=tokenizer, temperature=script_args.temperature,
                          max_new_tokens=script_args.max_new_tokens, device=local_rank)
    summarizer.model = deepspeed.init_inference(
        summarizer.model,
        mp_size=world_size,
        dtype=torch.float16,
        max_tokens=script_args.max_new_tokens,
        replace_with_kernel_inject=True
    )

    results = []
    for i, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
        raw_out: str = compute_conversation(row, summarizer)
        parsed = raw_out.replace(config.LLAMA_S_HEADER.strip(), "").strip()
        results.append(parsed)

        if i % 100 == 0:
            print([parsed])

    test_df["predicted"] = results
    out_file = f"{script_args.output_name}-test-predicted.parquet"
    test_df.to_parquet(out_file, index=False)

