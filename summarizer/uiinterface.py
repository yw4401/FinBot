from functools import partial

import asyncio
from langchain.chat_models import ChatVertexAI
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores.elasticsearch import ElasticsearchStore
from summarizer import config
from summarizer.topic_sum import ElasticSearchTopicRetriever, topic_aggregate_chain
import nest_asyncio

with open(config.ES_KEY_PATH, "r") as fp:
    es_key = fp.read().strip()
with open(config.ES_CLOUD_ID_PATH, "r") as fp:
    es_id = fp.read().strip()

nest_asyncio.apply()
embedding = SentenceTransformerEmbeddings(model_name=config.FILTER_EMBEDDINGS)
topic_store = ElasticsearchStore(index_name=config.ES_TOPIC_INDEX, embedding=embedding,
                                 es_cloud_id=es_id, es_api_key=es_key,
                                 vector_query_field=config.ES_TOPIC_VECTOR_FIELD,
                                 query_field=config.ES_TOPIC_FIELD)
article_store = ElasticsearchStore(index_name=config.ES_ARTICLE_INDEX, embedding=embedding,
                                   es_cloud_id=es_id, es_api_key=es_key,
                                   vector_query_field=config.ES_ARTICLE_VECTOR_FIELD,
                                   query_field=config.ES_ARTICLE_FIELD)
retriever = ElasticSearchTopicRetriever(topic_elasticstore=topic_store, chunks_elasticstore=article_store)
plan_llm = ChatVertexAI(
    project=config.GCP_PROJECT,
    temperature=0,
    model_name="chat-bison",
    max_output_tokens=512
)


async def get_qa_result(query):
    qa_agg_chain = topic_aggregate_chain(plan_llm, retriever, return_source_documents=True, verbose=True)
    summaries = [{"title": "Main Point", "keypoints": ["K1", "K2", "K3"]},
                 {"title": "Lorem Ipsum", "keypoints": ["K1", "K2", "K3"]}]
    qa_task = qa_agg_chain.acall(query)
    completed = await asyncio.gather(qa_task)

    return {
        "qa": completed[0]["result"].strip(),
        "summaries": summaries
    }


def finbot_response(text, period):
    return asyncio.run(get_qa_result(text))


if __name__ == "__main__":
    print(finbot_response(
        "Which companies are developing drugs that are currently in the process of approval or just got approved?",
        "1y"))