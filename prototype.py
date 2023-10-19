import logging
import sys
from pathlib import Path
import streamlit as st
from summarizer.ner import *

from summarizer.topic_sum import *
import summarizer.config
import pipeline.config

logging.basicConfig(level=logging.WARN, stream=sys.stdout)

#
# topics_summary_file can be acquired from gs://scraped-news-article-data-null/2023-topics-openai.parquet
# open_ai_api_file can be acquired from gs://scraped-news-article-data-null/apikey
# persist_dir_base and faiss_path can be acquired from gs://scraped-news-article-data-null/demo_setup.tar.xz.

# For the persist_dir_base and faiss_path, extract the archive, and the .index file is the faiss path. The directory
# is the persist_dir_base.

# To run the code, adjust the following variables appropriately so that they point to the right location.

year = 2023
month = 4

article_dir = str(Path(summarizer.config.TOPIC_ARTICLES_INDICES_DIR,
                       summarizer.config.TOPIC_ARTICLES_INDICES_PATTERN.format(year=year, month=month)))
topic_dir = str(Path(summarizer.config.TOPIC_SUMMARY_INDEX_DIR,
                     summarizer.config.TOPIC_SUMMARY_INDEX_PATTERN.format(year=year, month=month)))
chroma = chromadb.PersistentClient(path="topics/topic_indices/topic-2023-4")
chroma_articles = chromadb.PersistentClient(path="topics/topic_indices/articles-2023-4")
topic_retriever = TopicRetriever(client=chroma, collection=config.TOPIC_COLLECTION,
                                 embedding=config.TOPIC_EMBEDDING)
plan_llm = ChatVertexAI(
    temperature=0,
    model_name="chat-bison",
    max_output_tokens=512
)
filter_chain = create_topic_filter(plan_llm, topic_retriever)

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
        company = extract_company_ticker("")
        # Do whatever you need for API etc
        st.write(company)
        with st.spinner('Finding Relevant Topic...'):
            topics = filter_chain(query)

        with st.spinner("Synthesizing Response"):
            agg_chain = topic_aggregate_chain(plan_llm, chroma_articles, topics, return_source_documents=True)
            result = agg_chain(query)
            st.write(result["result"])
    else:
        st.write("Put something in the inquiry and submit!")
