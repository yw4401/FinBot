import math
import re

import pandas as pd
import spacy
import torch
from allennlp.predictors.predictor import Predictor
from datasets import load_dataset

import config
from coref_resolve import add_coreference_resolution


def extract_context(c):
    extract_regex = re.compile(r"Please answer the given financial question based on the context.\n"
                               r"Context: (?P<context>(.|\n)+)\nQuestion:")
    match = extract_regex.match(c)
    if not match:
        print(c)
        raise ValueError()
    else:
        return match.group("context")


def load_finqa():
    finqa_ds = load_dataset(path="ChanceFocus/flare-finqa", split="train")
    finqa = pd.DataFrame(finqa_ds)
    finqa = finqa.rename(columns={"query": "context", "text": "question"})
    finqa["context"] = finqa.context.apply(lambda c: extract_context(c))
    return finqa[["question", "context", "answer"]]


def load_webglm():
    webglm_ds = load_dataset("THUDM/webglm-qa", split="train")
    webglm_df = pd.DataFrame(webglm_ds)
    webglm_df["context"] = webglm_df.references.apply(lambda l: "\n".join(l))
    remove_regex = re.compile(r"\[\d+]")
    webglm_df["answer"] = webglm_df["answer"].apply(lambda a: remove_regex.sub("", a))
    return webglm_df[["question", "context", "answer"]]


if __name__ == "__main__":
    TAT_QA_URL = "gs://finetuningllama/tat_qa_rewritten.parquet"
    tat_qa_df = pd.read_parquet(TAT_QA_URL)[["question", "related_text", "answer"]].rename(
        columns={"related_text": "context"})
    web_glm_df = load_webglm()
    finace_related = tat_qa_df.shape[0]
    generic = 21000 - finace_related
    final_df = pd.concat([tat_qa_df, web_glm_df.sample(generic)], ignore_index=True)
    print("Final Size: " + str(final_df.shape[0]))

    predictor = Predictor.from_path(config.ARTICLE_COREF_MOD_URL, cuda_device=torch.cuda.current_device())
    nlp = spacy.load(config.ARTICLE_COREF_SPACY_MOD)
    coref_df = add_coreference_resolution(final_df, predictor=predictor, nlp=nlp, src_col="context").copy()
    coref_df["context"] = coref_df.coref
    coref_df = coref_df.drop("coref", axis=1)
    coref_df.to_parquet(f"gs://{config.FINE_TUNE_TARGET_BUCKET}/{config.FINE_TUNE_QA_COREF}", index=False)

