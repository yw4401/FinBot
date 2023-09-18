import logging
import sys
from pathlib import Path
from typing import Optional, List, Mapping, Any

import pandas as pd
import streamlit as st
import torch
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms.base import LLM
from llama_index import LLMPredictor, PromptHelper
from llama_index import QuestionAnswerPrompt
from llama_index import ServiceContext, LangchainEmbedding
from llama_index import load_index_from_storage
from llama_index.prompts.prompts import SimpleInputPrompt
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.storage.index_store import SimpleIndexStore
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import SimpleVectorStore
from transformers import LlamaTokenizer, LlamaForCausalLM

from summarizer.topic_sum import create_topic_filter, load_faiss_topic_filter

logging.basicConfig(level=logging.WARN, stream=sys.stdout)

#
#topics_summary_file can be acquired from gs://scraped-news-article-data-null/2023-topics-openai.parquet
#open_ai_api_file can be acquired from gs://scraped-news-article-data-null/apikey
#persist_dir_base and faiss_path can be acquired from gs://scraped-news-article-data-null/demo_setup.tar.xz.

#For the persist_dir_base and faiss_path, extract the archive, and the .index file is the faiss path. The directory
#is the persist_dir_base.

#To run the code, adjust the following variables appropriately so that they point to the right location.

topics_summary_file = "/home/sdai/Documents/ConversationAI/2023-topics-openai.parquet"
open_ai_api_file = "/home/sdai/Documents/ConversationAI/apikey"
persist_dir_base = "/home/sdai/Documents/ConversationAI/topic_indices"
faiss_path = "/home/sdai/Documents/ConversationAI/faiss_topic.index"

OPENAI_FINAL_PROMPT = "Summarize the provided information such that it answers the given inquiry and only the given inquiry. " + \
                      "The response should read well in addition to providing relevant and accurate information." + \
                      "Do not mention any time sensitive information such as share prices. " + \
                      'If the inquiry cannot be answered using only the provided information, ' + \
                      'write "CANNOT ANSWER" as the response. ' + \
                      'The response should be formatted in plain text.\n\n' + \
                      "Inquiry: {query_str}\n\nInformation:\n{context_str}\n"

topic_df = pd.read_parquet(topics_summary_file)
topic_existing_sum = topic_df.loc[
    (topic_df.summary.str.len() > 0) & (topic_df.summary.str.lower().str.strip() != "no theme")]
with open(open_ai_api_file, "r") as api_fp:
    api_key = api_fp.read().strip()

filter_llm = create_topic_filter(api_key=api_key, temperature=0)
embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))


class LlamaBased(LLM):
    tokenizer: LlamaTokenizer
    model: LlamaForCausalLM
    max_new_tokens: int
    query_helper: SimpleInputPrompt

    def __init__(self, model_path, max_new_tokens, query_helper):
        tokenizer = LlamaTokenizer.from_pretrained(model_path, device_map="auto", max_input_size=2048)
        model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
        LLM.__init__(self, tokenizer=tokenizer, model=model, max_new_tokens=max_new_tokens, query_helper=query_helper)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        prompt = self.query_helper.format(query_str=prompt)
        logging.info("final prompt: " + prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                      do_sample=False,
                                      temperature=0)
        result = self.tokenizer.batch_decode(outputs, skip_special_tokens=True, spaces_between_special_tokens=False)[0]
        # only return newly generated tokens
        return result[len(prompt):]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": "Koala"}

    @property
    def _llm_type(self) -> str:
        return "llama"


def create_llama_service_context(embed_model, model_path, query_wrapper_prompt, num_output=512, max_chunk_overlap=20):
    llm_predictor = LLMPredictor(llm=LlamaBased(model_path, num_output, query_wrapper_prompt))
    prompt_helper = PromptHelper(max_input_size=2048, num_output=num_output, max_chunk_overlap=max_chunk_overlap)
    service_context = ServiceContext.from_defaults(chunk_size_limit=512, llm_predictor=llm_predictor,
                                                   embed_model=embed_model, prompt_helper=prompt_helper)
    return service_context


def create_openai_service_context(embed_model, api_key, num_output=512, max_chunk_overlap=20):
    prompt_helper = PromptHelper(max_input_size=4096, num_output=num_output, max_chunk_overlap=max_chunk_overlap)
    service_context = ServiceContext.from_defaults(chunk_size_limit=512,
                                                   embed_model=embed_model,
                                                   prompt_helper=prompt_helper,
                                                   llm_predictor=LLMPredictor(llm=ChatOpenAI(model_name="gpt-3.5-turbo",
                                                                                             temperature=0,
                                                                                             openai_api_key=api_key))
                                                   )
    return service_context


def create_llama_index_lazyloader(service_context, directory):
    index_dict = {}

    def get_llama_index(topic_number):
        if topic_number not in index_dict:
            topic_index_dir = Path(directory, str(topic_number))
            if not topic_index_dir.exists():
                return None
            storage_context = StorageContext.from_defaults(
                docstore=SimpleDocumentStore.from_persist_dir(persist_dir=topic_index_dir),
                vector_store=SimpleVectorStore.from_persist_dir(persist_dir=topic_index_dir),
                index_store=SimpleIndexStore.from_persist_dir(persist_dir=topic_index_dir),
            )
            index = load_index_from_storage(storage_context=storage_context, service_context=service_context)
            index_dict[topic_number] = index
        return index_dict[topic_number]

    return get_llama_index


lazy_loader = create_llama_index_lazyloader(create_openai_service_context(embed_model, api_key), persist_dir_base)
faiss_topic_filter = load_faiss_topic_filter(faiss_path, embed_model=embed_model)

st.set_page_config(layout="wide")
left_column, right_column = st.columns([1, 2])

with left_column:
    st.write("# Welcome to FinBot")

    query = st.text_input("Inquiry")
    time = st.selectbox(
        'How far back do you want to go',
        ('Lorem Ipsum', '1 Week', '2 Week', '1 Month'))
    inquiry_submitted = st.button("Submit")

with right_column:
    if inquiry_submitted and query.strip() != "":
        QA_PROMPT = QuestionAnswerPrompt(OPENAI_FINAL_PROMPT)

        topic_filtered = faiss_topic_filter(query, topic_existing_sum)
        topic_generator = filter_llm(topic_filtered, query)
        while True:
            try:
                with st.spinner('Finding Relevant Topic...'):
                    r = next(topic_generator)
                    if r.rating < 0.5:
                        continue
                    index = lazy_loader(r.topic_number)
                    if not index:
                        continue
                    query_engine = index.as_query_engine(text_qa_template=QA_PROMPT)
                with st.spinner("Generating Response"):
                    result_str = query_engine.query(query)
                    if "CANNOT ANSWER" in result_str.response:
                        continue
                st.write(result_str.response)
            except StopIteration:
                break
    else:
        st.write("Put something in the inquiry and submit!")
