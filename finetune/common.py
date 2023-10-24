from typing import List

import torch
from transformers import (
    LlamaTokenizer,
)

import config

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."


class ListDataset(torch.utils.data.Dataset):

    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]


def token_count(text, tokenizer):
    tokenized = tokenizer.encode(text, add_special_tokens=False)
    return len(tokenized)


def restrict_article(text, max_token, tokenizer: LlamaTokenizer):
    tokenized = tokenizer.encode(text, add_special_tokens=False)
    if len(tokenized) > max_token:
        return tokenizer.decode(tokenized[:max_token], skip_special_tokens=False)
    return text


def format_prompt(examples):
    if examples[0]["role"] == "system":
        examples = [
                       {
                           "role": examples[1]["role"],
                           "content": B_SYS
                                      + examples[0]["content"]
                                      + E_SYS
                                      + examples[1]["content"],
                       }
                   ] + examples[2:]
    assert all([msg["role"] == "user" for msg in examples[::2]]) and all(
        [msg["role"] == "assistant" for msg in examples[1::2]]
    ), (
        "model only supports 'system', 'user' and 'assistant' roles, "
        "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
    )
    dialog_texts: List[str] = [
        f"<s>{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} </s>"
        for prompt, answer in zip(
            examples[::2],
            examples[1::2],
        )
    ]
    return "".join(dialog_texts)


def truncate_summary_example(question, body, summary, tokenizer, max_context):
    body_tokens = max_context - token_count(question, tokenizer) - token_count(summary, tokenizer) - 20
    return restrict_article(body, body_tokens, tokenizer)


def format_llama_sum_user(question, body):
    return f"{config.LLAMA_Q_HEADER}\n{question}\n\n{config.LLAMA_C_HEADER}\n{body}"


def format_llama_sum_resp(summary):
    return config.LLAMA_S_HEADER + summary


def format_summary_example(example):
    output_texts = []
    for i in range(len(example['body'])):
        q = example["question"][i]
        c = example["body"][i]
        s = format_llama_sum_resp(example["summary"][i])

        user = format_llama_sum_user(q, c)
        text = format_prompt([
            {"role": "system", "content": config.LLAMA_SUMMARY_BULLET_INSTRUCTION},
            {"role": "user", "content": user},
            {"role": "assistant", "content": s}
        ])
        output_texts.append(text[len("<s>"):-len("</s>")])

    return output_texts
