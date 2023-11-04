import asyncio
from typing import List

from langchain.chains import LLMChain
from langchain.output_parsers import CommaSeparatedListOutputParser, PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field

import summarizer.config as config
from summarizer.uiinterface import get_qa_llm


class KPIGroup(BaseModel):

    group_title: str = Field(description="The title of the logical group of KPIs")
    group_members: List[str] = Field(description="The list of KPIs belonging to this group")


class RelevantKPI(BaseModel):

    relevance: str = Field(description="Explanation of why the picked KPIs are relevant")
    groups: List[KPIGroup] = Field(description="Logical groups of relevant KPIs")


def get_ner_llm(kind=config.NER_MODEL, max_token=256):
    return get_qa_llm(kind, max_token)


def get_kpi_llm(kind=config.KPI_MODEL, max_token=512):
    return get_qa_llm(kind, max_token)


def build_ticker_extraction_chain(llm):
    prompt = PromptTemplate.from_template(config.NER_RESPONSE_PROMPT)
    output_parser = CommaSeparatedListOutputParser()
    prompt = prompt.partial(format_instructions=output_parser.get_format_instructions())
    llm_chain = LLMChain(llm=llm, prompt=prompt, output_parser=output_parser)
    return llm_chain


def format_response_for_ner(response):
    template = "{answer}\n{keypoints}"
    keypoints = sum([[k["title"]] + k["keypoints"] for k in response["summaries"]], [])
    return template.format(answer=response["qa"], keypoints=". ".join(keypoints))


def extract_company_ticker(query, response):
    llm = get_ner_llm()
    chain = build_ticker_extraction_chain(llm)
    result = asyncio.run(chain.arun({"text": response, "query": query}))
    normed_result = []
    for r in result:
        if "." in r:
            normed_result.append(r)
            normed_result.append(r.split(".")[0].strip())
        else:
            normed_result.append(r)
    return normed_result


def build_kpi_extraction_chain(llm):
    prompt = PromptTemplate.from_template(config.KPI_PROMPT)
    output_parser = PydanticOutputParser(pydantic_object=RelevantKPI)
    prompt = prompt.partial(format_instructions=output_parser.get_format_instructions())
    llm_chain = LLMChain(llm=llm, prompt=prompt, output_parser=output_parser)
    return llm_chain


def extract_relevant_field(query, response, ticker):
    llm = get_kpi_llm()
    chain = build_kpi_extraction_chain(llm)
    kpi_list = ", ".join(ticker.info.keys())
    return asyncio.run(chain.arun({"query": query, "response": response, "kpi": kpi_list}))


def extract_industry(text):
    return ""


def find_major_firms(industry):
    return []
