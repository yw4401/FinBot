import datetime
import time

import asyncio
from typing import List

import config
import google.cloud.bigquery as bq
import google.cloud.bigquery.dbapi as bqapi
import numpy as np
import pandas as pd
import re
import tqdm
import tqdm.asyncio as tqao
from asyncio import Semaphore
from contextlib import closing
from google.api_core.exceptions import InvalidArgument, ResourceExhausted, InternalServerError
from google.oauth2 import service_account
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.chat_models import ChatOpenAI, ChatVertexAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage
from openai.error import RateLimitError, Timeout, APIConnectionError, APIError
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split

tqdm.tqdm.pandas()


class AugmentedOutput(BaseModel):
    answerable: str = Field(description="The answerable question", default="FAILED")
    unanswerable: str = Field(description="The unanswerable question", default="FAILED")


class FilteredOutput(BaseModel):
    relevant: bool = Field(description="Whether the summary is relevant to investor", default=False)
    movement: bool = Field(description="Whether the summary talks about the movement of share prices, index, "
                                       "or information that can easily be extracted from a stock screener.",
                           default=False)


def create_full_prompt(system, user, examples):
    final_examples = []
    for e in examples:
        final_examples.append(HumanMessage(content=e["user"]))
        final_examples.append(AIMessage(content=e["assistant"]))
    chat_prompt = ChatPromptTemplate.from_messages([system, *final_examples, user])
    return chat_prompt


def create_summary_aug_chain(llm):
    system_prompt = SystemMessagePromptTemplate.from_template(config.SUMMARY_AUG_SYSTEM)
    user_prompt = HumanMessagePromptTemplate.from_template(config.SUMMARY_AUG_USER)
    chat_prompt = create_full_prompt(system_prompt, user_prompt, config.SUMMARY_AUG_EXAMPLES)
    augment_chain = LLMChain(llm=llm, prompt=chat_prompt)
    return augment_chain


def create_aug_parse_chain(llm):
    output_parser = PydanticOutputParser(pydantic_object=AugmentedOutput)
    system_prompt = SystemMessagePromptTemplate.from_template(config.SUMMARY_AUG_PARSE_SYSTEM)
    user_prompt = HumanMessagePromptTemplate.from_template(config.SUMMARY_AUG_PARSE_USER)
    chat_prompt = create_full_prompt(system_prompt, user_prompt, config.SUMMARY_AUG_PARSE_EXAMPLE)
    chat_prompt = chat_prompt.partial(format_instructions=output_parser.get_format_instructions())
    parse_chain = LLMChain(llm=llm, prompt=chat_prompt, output_parser=output_parser)
    return parse_chain


def create_rewrite_reltime_chain(llm):
    system_prompt = SystemMessagePromptTemplate.from_template(config.SUMMARY_REWRITE_SYSTEM)
    user_prompt = HumanMessagePromptTemplate.from_template(config.SUMMARY_REWRITE_USER)
    chat_prompt = create_full_prompt(system_prompt, user_prompt, config.SUMMARY_REWRITE_EXAMPLES)
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
    chat_prompt = create_full_prompt(system_prompt, user_prompt, config.SUMMARY_FILTER_EXAMPLE)
    filter_chain = LLMChain(llm=llm, prompt=chat_prompt)
    return filter_chain


def create_rewrite_headline_chain(llm):
    system_prompt = SystemMessagePromptTemplate.from_template(config.SUMMARY_SUMMARY_SYSTEM)
    user_prompt = HumanMessagePromptTemplate.from_template(config.SUMMARY_SUMMARY_USER)
    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
    augment_chain = LLMChain(llm=llm, prompt=chat_prompt)
    return augment_chain


def get_llm(kind="openai"):
    if kind == "openai":
        with open(config.OPENAI_KEY_PATH) as fp:
            key = fp.read().strip()

        plan_llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=key,
                              temperature=0, max_tokens=1024, max_retries=1, request_timeout=10)
        return plan_llm
    elif kind == "vertexai":
        credentials = service_account.Credentials.from_service_account_file(config.VERTEX_AI_KEY_PATH)
        plan_llm = ChatVertexAI(temperature=config.SUMMARY_AUG_TEMPERATURE,
                                model_name=config.SUMMARY_AUG_MODEL,
                                project=config.VERTEX_AI_PROJECT,
                                credentials=credentials,
                                max_output_tokens=1024)
        return plan_llm
    else:
        raise NotImplemented()


async def async_retry_with_backoff(func, *params, limiter=None, start=1, factor=2, max_retry=6,
                                   exceptions=(Timeout, RateLimitError, APIConnectionError, APIError)):
    retry_time = 0
    while True:
        if retry_time > max_retry:
            raise ValueError()
        try:
            if limiter:
                async with limiter:
                    return await func(*params)
            else:
                return await func(*params)
        except exceptions:
            amount = start * factor ** retry_time
            delta = amount * 0.2
            noise = (amount - delta) + np.random.random() * delta * 2
            await asyncio.sleep(noise)


async def filter_summary_complete(summary, llm, limiter):
    filter_chain = create_summary_filter_text_chain(llm)
    parse = create_summary_filter_parse_chain(llm)
    combined = SimpleSequentialChain(chains=[filter_chain, parse])
    try:
        return await async_retry_with_backoff(combined.acall, summary, limiter=limiter)
    except (ValueError, InvalidArgument):
        return {"output": FilteredOutput()}


async def augment_summary_complete(text, plan_llm, limiter):
    augment = create_summary_aug_chain(plan_llm)
    parse = create_aug_parse_chain(plan_llm)
    combined = SimpleSequentialChain(chains=[augment, parse])
    try:
        return await async_retry_with_backoff(combined.acall, text, limiter=limiter)
    except (ValueError, InvalidArgument):
        return {"output": AugmentedOutput()}


async def rewrite_summary_time(title, text, date, llm, limiter):
    rewriter = create_rewrite_reltime_chain(llm)
    date_str = date.strftime('%Y-%m-%d, %A')
    try:
        return await async_retry_with_backoff(rewriter.acall,
                                              {"ipt_text": text, "title": title, "date": date_str}, limiter=limiter)
    except (ValueError, InvalidArgument):
        return {"text": "FAILED"}


async def summary_summary_headline(text, llm, limiter):
    rewriter = create_rewrite_headline_chain(llm)
    try:
        return await async_retry_with_backoff(rewriter.acall, {"summary": text}, limiter=limiter)
    except (ValueError, InvalidArgument):
        return {"text": "FAILED"}


async def filter_summary(df, max_request=25):
    plan_llm = get_llm()
    relevant = []
    stat = []
    limiter = Semaphore(value=max_request)
    flist = [filter_summary_complete(i, plan_llm, limiter) for i in df.summary.tolist()]
    for i, augmentation in enumerate(await tqao.tqdm_asyncio.gather(*flist)):
        output = augmentation["output"]
        relevant.append(output.relevant)
        stat.append(output.movement)
    result = df.copy()
    result["relevant"] = relevant
    result["stat"] = stat
    return result.loc[result.relevant & ~result.stat]


async def rewrite_summary(df, max_request=25):
    plan_llm = get_llm()
    errors = []
    summaries = []
    limiter = Semaphore(value=max_request)
    flist = [rewrite_summary_time(row["title"], row["summary"],
                                  row["published"], plan_llm, limiter) for _, row in
             df.iterrows()]
    for i, augmentation in enumerate(await tqao.tqdm_asyncio.gather(*flist)):
        output = augmentation["text"]
        if "please send us your feedback" in output.lower() or "as a large language model" in output.lower():
            errors.append(i)
        else:
            summaries.append(output)
    result = df.copy()
    if len(errors) > 0:
        result = result.drop(index=result.iloc[errors].index)
    result["summary"] = summaries
    return result


async def headline_summary(df, max_request=25):
    plan_llm = get_llm()
    errors = []
    summaries = []
    limiter = Semaphore(value=max_request)
    flist = [summary_summary_headline(row["summary"], plan_llm, limiter) for _, row in df.iterrows()]
    for i, augmentation in enumerate(await tqao.tqdm_asyncio.gather(*flist)):
        output = augmentation["text"]
        if "please send us your feedback" in output.lower() or "as a large language model" in output.lower():
            errors.append(i)
        else:
            summaries.append(output)
    result = df.copy()
    if len(errors) > 0:
        result = result.drop(index=result.iloc[errors].index)
    result["headline"] = summaries
    result["summary"] = result["headline"].str.strip() + "\n" + result.summary.str.strip()
    result = result.drop("headline", axis=1)
    return result


async def augment_summary(df, max_request=25):
    plan_llm = get_llm()
    questions = []
    reverse_questions = []
    errors = []
    limiter = Semaphore(value=max_request)
    flist = [augment_summary_complete(i, plan_llm, limiter) for i in df.summary.tolist()]
    for i, augmentation in enumerate(await tqao.tqdm_asyncio.gather(*flist)):
        output = augmentation["output"]
        if output.unanswerable == "FAILED" or output.answerable == "FAILED":
            errors.append(i)
        else:
            questions.append(output.answerable)
            reverse_questions.append(output.unanswerable)
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


def inject_noise(df, splitter, target_chunks=7):
    df = df.reset_index(drop=True)
    chunks = df.body.progress_apply(splitter.split_text)
    chunks: List[List[str]] = [[c.strip() for c in chunk] for chunk in chunks]
    new_bodies = []
    for i, row in df.iterrows():
        published_date: datetime.datetime = row["published"]
        body_chunks = [f"Published: {published_date.strftime('%Y-%m-%d')}\n{c}" for c in chunks[i]]
        body_chunks = body_chunks[:min(target_chunks, len(body_chunks))]
        for _ in range(target_chunks - len(body_chunks)):
            alt_idx = np.random.choice(df.index)
            alternative_article = chunks[alt_idx]
            alt_date = df["published"].loc[alt_idx].strftime('%Y-%m-%d')
            alt_choice = np.random.choice(alternative_article)
            alt_text = f"Published: {alt_date}\n{alt_choice}"
            body_chunks.append(alt_text)
        np.random.shuffle(body_chunks)
        new_bodies.append("\n\n".join(body_chunks))
    df = df.copy()
    df["body"] = new_bodies
    return df
