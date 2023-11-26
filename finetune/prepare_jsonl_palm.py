import json
from dataclasses import dataclass, field
from typing import cast, Optional

import pandas as pd
from transformers import (
    HfArgumentParser
)

import config


@dataclass
class ScriptArguments:
    dataset_path: Optional[str] = field(default="./fine-tune-qa-train.parquet")
    output_path: Optional[str] = field(default="./fine-tune-qa-train.jsonl")
    sample: Optional[int] = field(default=50000)


def create_dict(row):
    return {
        "input_text": config.PALM_QA_PROMPT.format(context=row["context"], question=row["question"]),
        "output_text": row["answer"]
    }


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    script_args: ScriptArguments = cast(ScriptArguments, script_args)

    data_df = pd.read_parquet(script_args.dataset_path)
    if data_df.shape[0] > script_args.sample:
        data_df = data_df.sample(n=script_args.sample, random_state=93)
    print(data_df.shape[0])
    data_dicts = data_df.apply(create_dict, axis=1)

    out_strs = [json.dumps(d) for d in data_dicts]
    out = "\n".join(out_strs)
    with open(script_args.output_path, "w") as fp:
        fp.write(out)


if __name__ == "__main__":
    main()
