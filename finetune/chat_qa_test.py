from dataclasses import dataclass, field
from typing import cast, Optional

import pandas as pd
from transformers import (
    HfArgumentParser, AutoTokenizer
)
from vllm import SamplingParams, LLM

import config
from common import format_llama_qa_user, truncate_qa_example_chat


@dataclass
class ScriptArguments:
    model_path: Optional[str] = field(default="Open-Orca/Mistral-7B-OpenOrca")
    template_path: Optional[str] = field(default=None)
    output_name: Optional[str] = field(default="orca-mistral")
    dataset_path: Optional[str] = field(default="./fine-tune-qa-test.parquet")
    token_path: Optional[str] = field(default="./hf_token")
    dtype: Optional[str] = field(default="float16")
    world_size: Optional[int] = field(default=2)
    max_input_length: Optional[int] = field(default=4096 - 256)
    max_new_tokens: Optional[int] = field(default=256)
    temperature: Optional[float] = field(default=0)
    stop_criteria: Optional[str] = field(default=None)
    gpu_utilization: Optional[float] = field(default=0.9)


def prepare_conversation(row, tokenizer, chat_template=None):
    conversation = [{"role": "system", "content": config.LLAMA_QA_SYSTEM_INSTRUCTION},
                    {"role": "user", "content": format_llama_qa_user(dict(question=row["question"],
                                                                          context=row["context"]))}]
    chat_form = tokenizer.apply_chat_template(conversation, chat_template=chat_template,
                                              tokenize=False, add_generation_prompt=True)
    return chat_form


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    script_args: ScriptArguments = cast(ScriptArguments, script_args)
    template = None
    if script_args.template_path:
        with open(script_args.template_path, "r") as fp:
            template = fp.read()

    test_df = pd.read_parquet(script_args.dataset_path)
    with open(script_args.token_path, "r") as fp:
        hf_token = fp.read().strip()

    if script_args.stop_criteria:
        stops = script_args.stop_criteria.split(",")
    else:
        stops = []

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_path)
    sampling_params = SamplingParams(temperature=0, max_tokens=script_args.max_new_tokens, stop=stops)
    llm = LLM(model=script_args.model_path, dtype=script_args.dtype, gpu_memory_utilization=script_args.gpu_utilization,
              tensor_parallel_size=script_args.world_size)

    test_df["context"] = test_df.apply(
        lambda row: truncate_qa_example_chat(system=config.LLAMA_QA_SYSTEM_INSTRUCTION,
                                             question=row["question"],
                                             context=row["context"],
                                             answer=row["answer"],
                                             tokenizer=tokenizer,
                                             max_context=script_args.max_input_length), axis=1)
    raw_inputs = test_df.apply(lambda row: prepare_conversation(row, tokenizer, chat_template=template),
                               axis=1).tolist()
    results = []
    for i, output in enumerate(llm.generate(raw_inputs, sampling_params)):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        results.append(generated_text)
    test_df["predicted"] = results
    out_file = f"{script_args.output_name}-qa-test-predicted.parquet"
    test_df.to_parquet(out_file, index=False)
