import asyncio
from typing import List

from langchain.chains import LLMChain
from langchain.output_parsers import CommaSeparatedListOutputParser, PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field

import summarizer.config as config
from summarizer.uiinterface import get_qa_llm


class KPIGroup(BaseModel):
    """
    A group of KPI with a specified title.
    """

    group_title: str = Field(description="The title of the logical group of KPIs")
    group_members: List[str] = Field(description="The list of KPIs belonging to this group")


class RelevantKPI(BaseModel):
    """
    A list of relevant KPIs with the reason of relevance.
    """

    relevance: str = Field(description="Explanation of why the picked KPIs are relevant")
    groups: List[KPIGroup] = Field(description="Logical groups of relevant KPIs")


def get_ner_llm(kind=config.NER_MODEL, max_token=256):
    """
    gets a Langchain LLM for identifying ticker symbol.
    """
    return get_qa_llm(kind, max_token)


def get_kpi_llm(kind=config.KPI_MODEL, max_token=512):
    """
    gets a Langchain LLM for identifying relevant KPIs.
    """
    return get_qa_llm(kind, max_token)


def build_ticker_extraction_chain(llm):
    """
    Builds a langchain for extracting relevant ticker symbols.

    :param llm: the langchain LLM for the chain.
    :returns: the built chain that extracts ticker symbols as a list
    """

    prompt = PromptTemplate.from_template(config.NER_RESPONSE_PROMPT)
    output_parser = CommaSeparatedListOutputParser()
    prompt = prompt.partial(format_instructions=output_parser.get_format_instructions())
    llm_chain = LLMChain(llm=llm, prompt=prompt, output_parser=output_parser)
    return llm_chain


def format_response_for_ner(response):
    """
    Formats the string used to extract ticker symbols from the response dict.

    :param response: the response dict from qa and summarization
    :returns: a single string that can be used with extract_company_ticker
    """

    template = "{answer}\n{keypoints}"
    keypoints = sum([[k["title"]] + k["keypoints"] for k in response["summaries"]], [])
    return template.format(answer=response["qa"], keypoints=". ".join(keypoints))


def extract_company_ticker(query, response):
    """
    extracts a list of relevant ticker symbols given a query and response

    :param query: the input query from the user
    :param response: a response string containing the body of the response
    :returns: a list of relevant ticker symbols
    """

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
    """
    builds a langchain for identifying relevant KPIs

    :param llm: the langchain llm used for the chain
    """

    prompt = PromptTemplate.from_template(config.KPI_PROMPT)
    output_parser = PydanticOutputParser(pydantic_object=RelevantKPI)
    prompt = prompt.partial(format_instructions=output_parser.get_format_instructions())
    llm_chain = LLMChain(llm=llm, prompt=prompt, output_parser=output_parser)
    return llm_chain


def extract_relevant_field(query, response, ticker):
    """
    identify relevant KPIs in groups based on the query, response, and kpis available from the ticker

    :param query: the user query
    :param response: additional context for the query
    :param ticker: the YFinance ticker with .info field containing the KPIs
    :returns: RelevantKPI containing the relevant KPI groups
    """
    llm = get_kpi_llm()
    chain = build_kpi_extraction_chain(llm)
    kpi_str = ""
    for k, v in ticker.info.items():
        try:
            kpi_str = f"{kpi_str}\n{k}: {float(v)}"
        except (ValueError, TypeError):
            pass

    try:
        return asyncio.run(chain.arun({"query": query, "response": response, "kpi": kpi_str.strip()}))
    except Exception:
        return RelevantKPI(relevance="None", groups=[])


def extract_industry(text):
    return ""


def find_major_firms(industry):
    return []
