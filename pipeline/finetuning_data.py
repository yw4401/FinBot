import re
from contextlib import closing
import pandas as pd
from langchain.chains import LLMChain
from langchain.chat_models import ChatVertexAI
from langchain.output_parsers import RegexParser
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import joblib

import google.cloud.storage as storage

import config
from select_data import load_subsample_index

prefix_bullets = "context: {body}\n\nquestion: {question}\n\nsummarize the key-points to answer the question: "
prefix_plain = "context: {body}\n\nquestion: {question}\n\nsummarize to answer the question: "


def prepend_prefix(row):
    if row["summary_type"] == "BULLETS":
        return prefix_bullets.format(body=row["body"], question=row["question"])
    elif row["summary_type"] == "PLAIN":
        return prefix_plain.format(body=row["body"], question=row["question"])
    else:
        raise ValueError("Type")


def create_summary_aug_chain(llm):
    output_parser = RegexParser(regex=config.SUMMARY_AUG_REGEX, output_keys=["question"])
    system_prompt = SystemMessagePromptTemplate.from_template(config.SUMMARY_AUG_SYSTEM)
    user_prompt = HumanMessagePromptTemplate.from_template(config.SUMMARY_AUG_USER)
    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
    augment_chain = LLMChain(llm=llm, prompt=chat_prompt, output_parser=output_parser)
    return augment_chain


def augment_summary_complete(text):
    plan_llm = ChatVertexAI(temperature=config.SUMMARY_AUG_TEMPERATURE,
                            model_name=config.SUMMARY_AUG_MODEL, max_output_tokens=1024)
    chain = create_summary_aug_chain(plan_llm)
    return chain(text)


def augment_summary(df, jobs=3):
    questions = []
    parallel = joblib.Parallel(n_jobs=jobs, return_as="generator")
    with tqdm(total=df.shape[0]) as progress_bar:
        for augmentation in parallel(joblib.delayed(augment_summary_complete)(i) for i in df.summary.tolist()):
            questions.append(augmentation["text"]["question"].strip())
            progress_bar.update(1)
    result = df.copy()
    result["question"] = questions
    return result


def get_data_sets_df(sample_df, test_instances=1000):
    sample_df = sample_df.sample(frac=1, random_state=93).reset_index(drop=True)
    clean_regex = re.compile(r"\*[\s\n]*(?=\*)")
    sample_df["summary"] = sample_df.summary.apply(lambda s: clean_regex.sub(" ", s).strip())
    sample_df["summary"] = sample_df.title.str.strip() + "\n" + sample_df.summary
    sample_df["summary"] = sample_df.summary.str.strip()
    sample_df["body"] = sample_df.apply(prepend_prefix, axis=1)
    sample_df = sample_df[["body", "summary"]]

    train_df, test_df = train_test_split(sample_df, test_size=test_instances)
    return train_df, test_df


if __name__ == "__main__":

    articles = []
    with closing(storage.Client(project=config.GCP_PROJECT)) as client:
        idx = load_subsample_index(client=client)
    target_url = "gs://{bucket}/{pattern}".format(bucket=config.FINE_TUNE_TARGET_BUCKET,
                                                  pattern=config.FINE_TUNE_FILE_PATTERN)
    with tqdm(total=len(idx["files"])) as progress:
        for f in idx["files"]:
            temp_df = pd.read_parquet(f)
            articles.append(temp_df)
            progress.update(1)

    df = pd.concat(articles, ignore_index=True)
    df = df.loc[df.summary_type != "NULL"]
    df = df.drop_duplicates(subset=["body"])
    df = augment_summary(df)
    print(df.head())
    print("Uploading article snapshots")
    train, test = get_data_sets_df(df)
    train.to_parquet(target_url.format(split="train"), index=False)
    test.to_parquet(target_url.format(split="test"), index=False)
