import re
import time
from contextlib import closing

import google.cloud.bigquery as bq
import google.cloud.bigquery.dbapi as bqapi
import joblib
import pandas as pd
from google.api_core.exceptions import InvalidArgument, ResourceExhausted, InternalServerError
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.chat_models import ChatVertexAI, ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

import config
from pipeline.build_vector_index import create_splitter

tqdm.pandas()


class AugmentedOutput(BaseModel):
    answerable: str = Field(description="The answerable question", default="FAILED")
    unanswerable: str = Field(description="The unanswerable question", default="FAILED")


class FilteredOutput(BaseModel):
    relevant: bool = Field(description="Whether the summary is relevant to investor", default=False)
    movement: bool = Field(description="Whether the summary talks about the movement of share prices, index, "
                                       "or information that can easily be extracted from a stock screener.",
                           default=False)


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


def create_rewrite_reltime_chain(llm):
    system_prompt = SystemMessagePromptTemplate.from_template(config.SUMMARY_REWRITE_SYSTEM)
    user_prompt = HumanMessagePromptTemplate.from_template(config.SUMMARY_REWRITE_USER)
    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
    augment_chain = LLMChain(llm=llm, prompt=chat_prompt)
    return augment_chain


def create_summary_filter_parse_chain(llm):
    output_parser = PydanticOutputParser(pydantic_object=FilteredOutput)
    system_prompt = SystemMessagePromptTemplate.from_template(config.SUMMARY_AUG_PARSE_SYSTEM)
    user_prompt = HumanMessagePromptTemplate.from_template(config.SUMMARY_AUG_PARSE_USER)
    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
    chat_prompt = chat_prompt.partial(format_instructions=output_parser.get_format_instructions())
    parse_chain = LLMChain(llm=llm, prompt=chat_prompt, output_parser=output_parser)
    return parse_chain


def create_summary_filter_text_chain(llm):
    system_prompt = SystemMessagePromptTemplate.from_template(config.SUMMARY_FILTER_SYSTEM)
    user_prompt = HumanMessagePromptTemplate.from_template(config.SUMMARY_FILTER_USER)
    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
    filter_chain = LLMChain(llm=llm, prompt=chat_prompt)
    return filter_chain


def create_rewrite_headline_chain(llm):
    system_prompt = SystemMessagePromptTemplate.from_template(config.SUMMARY_SUMMARY_SYSTEM)
    user_prompt = HumanMessagePromptTemplate.from_template(config.SUMMARY_SUMMARY_USER)
    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
    augment_chain = LLMChain(llm=llm, prompt=chat_prompt)
    return augment_chain


def filter_summary_complete(summary, llm):
    filter_chain = create_summary_filter_text_chain(llm)
    parse = create_summary_filter_parse_chain(llm)
    combined = SimpleSequentialChain(chains=[filter_chain, parse])
    try:
        while True:
            try:
                return combined(summary)
            except (ResourceExhausted, InternalServerError):
                time.sleep(1)
    except (ValueError, InvalidArgument):
        return {"output": FilteredOutput()}


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


def rewrite_summary_time(title, text, date, llm):
    rewriter = create_rewrite_reltime_chain(llm)
    date_str = date.strftime('%Y-%m-%d, %A')
    try:
        while True:
            try:
                return rewriter({"ipt_text": text, "title": title, "date": date_str})
            except (ResourceExhausted, InternalServerError):
                time.sleep(1)
    except (ValueError, InvalidArgument):
        return {"text": "FAILED"}


def summary_summary_headline(text, llm):
    rewriter = create_rewrite_headline_chain(llm)
    try:
        while True:
            try:
                return rewriter({"summary": text})
            except (ResourceExhausted, InternalServerError):
                time.sleep(1)
    except (ValueError, InvalidArgument):
        return {"text": "FAILED"}


def filter_summary(df, jobs=5):
    plan_llm = ChatVertexAI(temperature=config.SUMMARY_AUG_TEMPERATURE,
                            model_name=config.SUMMARY_AUG_MODEL,
                            project=config.GCP_PROJECT,
                            max_output_tokens=1024)
    relevant = []
    stat = []
    parallel = joblib.Parallel(n_jobs=jobs, return_as="generator", backend="threading")
    with tqdm(total=df.shape[0]) as progress_bar:
        for i, augmentation in enumerate(
                parallel(joblib.delayed(filter_summary_complete)(i, plan_llm) for i in df.summary.tolist())):
            output = augmentation["output"]
            relevant.append(output.relevant)
            stat.append(output.movement)
            progress_bar.update(1)
    result = df.copy()
    result["relevant"] = relevant
    result["stat"] = stat
    return result.loc[result.relevant & ~result.stat]


def rewrite_summary(df, jobs=5):
    with open(config.OPENAI_KEY_PATH) as fp:
        key = fp.read().strip()

    plan_llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=key, temperature=0, max_tokens=1024)
    # credentials = service_account.Credentials.from_service_account_file(config.VERTEX_AI_KEY_PATH)
    # plan_llm = ChatVertexAI(temperature=config.SUMMARY_AUG_TEMPERATURE,
    #                         model_name=config.SUMMARY_AUG_MODEL,
    #                         project=config.VERTEX_AI_PROJECT,
    #                         credentials=credentials,
    #                         max_output_tokens=1024)
    errors = []
    summaries = []
    parallel = joblib.Parallel(n_jobs=jobs, return_as="generator", backend="threading")
    with tqdm(total=df.shape[0]) as progress_bar:
        for i, augmentation in enumerate(
                parallel(joblib.delayed(rewrite_summary_time)(row["title"], row["summary"],
                                                              row["published"], plan_llm) for _, row in
                         df.iterrows())):
            output = augmentation["text"]
            if "please send us your feedback" in output.lower():
                output = "FAILED"
            if output == "FAILED":
                errors.append(i)
            else:
                summaries.append(output)
            progress_bar.update(1)
    result = df.copy()
    if len(errors) > 0:
        result = result.drop(index=result.iloc[errors].index)
    result["summary"] = summaries
    return result


def headline_summary(df, jobs=5):
    plan_llm = ChatVertexAI(temperature=config.SUMMARY_AUG_TEMPERATURE,
                            model_name=config.SUMMARY_AUG_MODEL,
                            project=config.GCP_PROJECT,
                            max_output_tokens=1024)
    errors = []
    summaries = []
    parallel = joblib.Parallel(n_jobs=jobs, return_as="generator", backend="threading")
    with tqdm(total=df.shape[0]) as progress_bar:
        for i, augmentation in enumerate(
                parallel(joblib.delayed(summary_summary_headline)(row["summary"], plan_llm) for _, row in
                         df.iterrows())):
            output = augmentation["text"]
            if "please send us your feedback" in output.lower():
                output = "FAILED"
            if output == "FAILED":
                errors.append(i)
            else:
                summaries.append(output)
            progress_bar.update(1)
    result = df.copy()
    if len(errors) > 0:
        result = result.drop(index=result.iloc[errors].index)
    result["headline"] = summaries
    result["summary"] = result["headline"].str.strip() + "\n" + result.summary.str.strip()
    result = result.drop("headline", axis=1)
    return result


def augment_summary(df, jobs=5):
    plan_llm = ChatVertexAI(temperature=config.SUMMARY_AUG_TEMPERATURE,
                            model_name=config.SUMMARY_AUG_MODEL,
                            project=config.GCP_PROJECT,
                            max_output_tokens=1024)
    questions = []
    reverse_questions = []
    errors = []
    parallel = joblib.Parallel(n_jobs=jobs, return_as="generator", backend="threading")
    with tqdm(total=df.shape[0]) as progress_bar:
        for i, augmentation in enumerate(
                parallel(joblib.delayed(augment_summary_complete)(i, plan_llm) for i in df.summary.tolist())):
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
        question.extend([row["question"], row["reverse_question"]])
        pos.extend([True, False])
    result_df = pd.DataFrame({
        "body": body,
        "question": question,
        "summary": summary,
        "pos": pos
    })

    train_df, test_df = train_test_split(result_df, test_size=test_instances, stratify=result_df.pos)
    return train_df[["body", "question", "summary"]], test_df[["body", "question", "summary"]]


def get_full_data(client: bq.Client):
    query = "SELECT id, published, title, body, summary_type, summary FROM Articles.ScrapedArticles " \
            "WHERE summary_type = 'BULLETS'"
    with closing(bqapi.Connection(client=client)) as connection:
        with closing(connection.cursor()) as cursor:
            cursor.execute(query)
            results = [list(r) for r in cursor.fetchall()]
    return pd.DataFrame(results, columns=["id", "published", "title", "body", "summary_type", "summary"])


def inject_noise(df, splitter, target_chunks=8):
    chunks = df.body.progress_apply(splitter.split_text)
    new_bodies = []
    for i, row in df.iterrows():
        body_chunks = chunks[i]
        body_chunks = body_chunks[:min(target_chunks, len(body_chunks))]
        for i in range(target_chunks - len(body_chunks)):
            alternative_article = np.random.choice(chunks)
            body_chunks.append(np.random.choice(alternative_article))
        np.random.shuffle(body_chunks)
        new_bodies.append("".join(body_chunks))
    df = df.copy()
    df["body"] = new_bodies
    return df


if __name__ == "__main__":
    with bq.Client(project=config.GCP_PROJECT) as client:
        df = get_full_data(client)
        df = df.drop_duplicates(subset=["body"]).sample(10, random_state=93)
        print(df.summary.tolist())
        df = filter_summary(df)
        print(df.summary.tolist())
        df = rewrite_summary(df)
        print(df.summary.tolist())
        df = headline_summary(df)
        print(df.summary.tolist())
        df = augment_summary(df)
        print(df.head())
        df = inject_noise(df, create_splitter())
        print(df.body.tolist()[0])
    # print("Uploading article snapshots")
    # train, test = get_data_sets_df(df)
    # train.to_parquet(target_url.format(split="train"), index=False)
    # test.to_parquet(target_url.format(split="test"), index=False)
