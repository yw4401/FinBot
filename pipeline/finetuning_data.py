import re
import time
from contextlib import closing
import pandas as pd
from google.api_core.exceptions import InvalidArgument, ResourceExhausted, InternalServerError
from google.oauth2 import service_account
from langchain.chains import LLMChain, SequentialChain, SimpleSequentialChain
from langchain.chat_models import ChatVertexAI
from langchain.output_parsers import RegexParser, PydanticOutputParser
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import joblib

import google.cloud.storage as storage

import config
from select_data_2 import load_subsample_index


class AugmentedOutput(BaseModel):

    answerable: str = Field(description="The answerable question", default="FAILED")
    unanswerable: str = Field(description="The unanswerable question", default="FAILED")


def create_summary_aug_chain(llm):
    system_prompt = SystemMessagePromptTemplate.from_template(config.SUMMARY_AUG_SYSTEM)
    user_prompt = HumanMessagePromptTemplate.from_template(config.SUMMARY_AUG_USER)
    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
    augment_chain = LLMChain(llm=llm, prompt=chat_prompt)
    return augment_chain


def create_aug_parse_chain(llm):
    output_parser = PydanticOutputParser(pydantic_object=AugmentedOutput)
    system_prompt = SystemMessagePromptTemplate.from_template(config.SUMMARY_AUG_PARSE_SYSTEM)
    user_prompt = HumanMessagePromptTemplate.from_template(config.SUMMARY_AUG_PARSE_USER)
    user_example = HumanMessage(content=config.SUMMARY_AUG_PARSE_EXAMPLE["raw"])
    ai_example = AIMessage(content=config.SUMMARY_AUG_PARSE_EXAMPLE["output"])
    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_example, ai_example, user_prompt])
    chat_prompt = chat_prompt.partial(format_instructions=output_parser.get_format_instructions())
    parse_chain = LLMChain(llm=llm, prompt=chat_prompt, output_parser=output_parser)
    return parse_chain


def augment_summary_complete(text, plan_llm):
    augment = create_summary_aug_chain(plan_llm)
    parse = create_aug_parse_chain(plan_llm)
    combined = SimpleSequentialChain(chains=[augment, parse])
    try:
        while True:
            try:
                return combined(text)
            except (ResourceExhausted, InternalServerError):
                time.sleep(1)
    except (ValueError, InvalidArgument):
        return {"output": AugmentedOutput()}


def augment_summary(df, jobs=5):
    credentials = service_account.Credentials.from_service_account_file(config.VERTEX_AI_KEY_PATH)
    plan_llm = ChatVertexAI(temperature=config.SUMMARY_AUG_TEMPERATURE,
                            model_name=config.SUMMARY_AUG_MODEL,
                            credentials=credentials,
                            project=config.VERTEX_AI_PROJECT,
                            max_output_tokens=1024)
    questions = []
    reverse_questions = []
    errors = []
    parallel = joblib.Parallel(n_jobs=jobs, return_as="generator", backend="threading")
    with tqdm(total=df.shape[0]) as progress_bar:
        for i, augmentation in enumerate(parallel(joblib.delayed(augment_summary_complete)(i, plan_llm) for i in df.summary.tolist())):
            output = augmentation["output"]
            if output.unanswerable == "FAILED" or output.answerable == "FAILED":
                errors.append(i)
            else:
                questions.append(output.answerable)
                reverse_questions.append(output.unanswerable)
            progress_bar.update(1)
    result = df.copy()
    if len(errors) > 0:
        result = result.drop(index=result.iloc[errors].index)
    result["question"] = questions
    result["reverse_question"] = reverse_questions
    return result


def get_data_sets_df(sample_df, test_instances=1000):
    sample_df = sample_df.sample(frac=1, random_state=93).reset_index(drop=True)
    clean_regex = re.compile(r"\*[\s\n]*(?=\*)")
    sample_df["summary"] = sample_df.summary.apply(lambda s: clean_regex.sub(" ", s).strip())
    sample_df["summary"] = sample_df.title.str.strip() + "\n" + sample_df.summary
    sample_df["summary"] = sample_df.summary.str.strip()

    body = []
    summary = []
    question = []
    pos = []
    for i, row in sample_df.iterrows():
        body.extend([row["body"], row["body"]])
        summary.extend([row["summary"], "Impossible to answer with given information"])
        pos.extend([True, False])
    result_df = pd.DataFrame({
        "body": body,
        "question": question,
        "summary": summary,
        "pos": pos
    })

    train_df, test_df = train_test_split(result_df, test_size=test_instances, stratify=result_df.pos)
    return train_df[["body", "question", "summary"]], test_df[["body", "question", "summary"]]


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
    df = df.loc[df.summary_type == "BULLETS"]
    df = df.drop_duplicates(subset=["body"])
    df = augment_summary(df)
    print(df.head())
    print("Uploading article snapshots")
    train, test = get_data_sets_df(df)
    train.to_parquet(target_url.format(split="train"), index=False)
    test.to_parquet(target_url.format(split="test"), index=False)
