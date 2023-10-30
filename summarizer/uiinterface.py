import asyncio
from langchain.chat_models import ChatVertexAI, ChatOpenAI
from langchain.llms import OpenAI
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores.elasticsearch import ElasticsearchStore
import summarizer.config as config
from summarizer.topic_sum import ElasticSearchTopicRetriever, topic_aggregate_chain, aget_summaries, afind_top_topics
import streamlit as st

with open(config.ES_KEY_PATH, "r") as fp:
    es_key = fp.read().strip()
with open(config.ES_CLOUD_ID_PATH, "r") as fp:
    es_id = fp.read().strip()


@st.cache_resource
def get_embeddings():
    embedding = SentenceTransformerEmbeddings(model_name=config.FILTER_EMBEDDINGS, model_kwargs={'device': 'cpu'})

    return embedding


@st.cache_resource
def get_topic_store():
    embedding = get_embeddings()
    topic_store = ElasticsearchStore(index_name=config.ES_TOPIC_INDEX, embedding=embedding,
                                     es_cloud_id=es_id, es_api_key=es_key,
                                     vector_query_field=config.ES_TOPIC_VECTOR_FIELD,
                                     query_field=config.ES_TOPIC_FIELD)
    return topic_store


@st.cache_resource
def get_article_store():
    embedding = get_embeddings()
    article_store = ElasticsearchStore(index_name=config.ES_ARTICLE_INDEX, embedding=embedding,
                                       es_cloud_id=es_id, es_api_key=es_key,
                                       vector_query_field=config.ES_ARTICLE_VECTOR_FIELD,
                                       query_field=config.ES_ARTICLE_FIELD)
    return article_store


def answer_question(query, topics):
    plan_llm = ChatVertexAI(
        project=config.GCP_PROJECT,
        temperature=0,
        model_name="chat-bison",
        max_output_tokens=512
    )
    retriever = ElasticSearchTopicRetriever(topic_elasticstore=get_topic_store(),
                                            chunks_elasticstore=get_article_store(),
                                            topics=topics)
    qa_agg_chain = topic_aggregate_chain(plan_llm, retriever, return_source_documents=True, verbose=True)
    return asyncio.create_task(qa_agg_chain.acall(query))


def find_summaries(query, topics):
    plan_llm = OpenAI(openai_api_base=config.SUM_API_SERVER, model_name=config.SUM_API_MODEL,
                      max_tokens=512, temperature=0, openai_api_key="EMPTY", verbose=True)
    return aget_summaries(query, topics, get_article_store(), plan_llm)


async def get_qa_result(query):
    topic_results = await afind_top_topics(get_topic_store(), query, k=config.TOPIC_SUM_K)
    topic_nums = [topic.metadata["topic"] for topic in topic_results]
    qa_coro = asyncio.ensure_future(answer_question(query, topic_nums))
    sum_coro = asyncio.ensure_future(find_summaries(query, topic_nums))

    qa_completed, summaries = await asyncio.gather(qa_coro, sum_coro)

    return {
        "qa": qa_completed["result"].strip(),
        "summaries": summaries
    }


def finbot_response(text, period):
    return asyncio.run(get_qa_result(text))
