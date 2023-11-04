import asyncio

from langchain.chains import LLMChain
from langchain.chat_models import ChatVertexAI
from langchain.llms.openai import OpenAI
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate

import summarizer.config as config


def get_ner_llm(kind=config.NER_MODEL, max_token=256):
    if kind == "vertexai":
        plan_llm = ChatVertexAI(
            project=config.GCP_PROJECT,
            temperature=0,
            model_name="chat-bison",
            max_output_tokens=max_token
        )
        return plan_llm
    elif kind == "openai":
        with open(config.OPENAI_API) as fp:
            key = fp.read().strip()

        plan_llm = OpenAI(model_name="gpt-3.5-turbo-instruct", openai_api_key=key,
                          temperature=0, max_tokens=max_token)
        return plan_llm
    else:
        raise NotImplemented()


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
    result = asyncio.run(chain.arun({"text": format_response_for_ner(response), "query": query}))
    return result


def extract_relevant_field(text):
    return []


def extract_industry(text):
    return ""


def find_major_firms(industry):
    return []
