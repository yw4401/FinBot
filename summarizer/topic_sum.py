import asyncio
import re

from langchain.callbacks.base import Callbacks
from typing import Any, List, Optional, Dict

from joblib import Parallel, delayed
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.chains import RetrievalQA, LLMChain, SimpleSequentialChain
from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from langchain.chat_models import ChatVertexAI
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.schema import Document, BaseRetriever
from langchain.vectorstores import Chroma, ElasticsearchStore

try:
    import config
except ModuleNotFoundError:
    import summarizer.config as config


async def afind_top_topics(vector_db, query, k=5):
    topic_results = await vector_db.amax_marginal_relevance_search(query, k=k)
    return topic_results


def find_top_topics(vector_db, query, k=5):
    return vector_db.max_marginal_relevance_search(query, k=k)


class ElasticSearchTopicRetriever(BaseRetriever):
    topic_elasticstore: ElasticsearchStore
    chunks_elasticstore: ElasticsearchStore
    topic_k: int = config.TOPIC_K
    chunk_k: int = config.ARTICLE_K

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        topic_results = find_top_topics(self.topic_elasticstore, query, k=self.topic_k)
        parallel = Parallel(n_jobs=self.topic_k, backend="threading", return_as="generator")
        result = []
        topic_nums = [topic.metadata["topic"] for topic in topic_results]
        for chunk in parallel(delayed(self._get_relevant_chunks)(t, query) for t in topic_nums):
            result.extend(chunk)
        return result

    async def aget_relevant_documents(self, query: str, *, callbacks: Callbacks = None, tags: Optional[
        List[str]] = None, metadata: Optional[Dict[str, Any]] = None, run_name: Optional[str] = None, **kwargs: Any):
        topic_results = await afind_top_topics(self.topic_elasticstore, query, k=self.topic_k)
        topic_nums = [topic.metadata["topic"] for topic in topic_results]
        coros = [self.chunks_elasticstore.amax_marginal_relevance_search(query=query, k=self.chunk_k,
                                                                         filter=[{"term": {
                                                                             "metadata.topic": t}}]) for t in
                 topic_nums]
        result = []
        for r in await asyncio.gather(*coros):
            result.extend(r)
        return result

    def _get_relevant_chunks(self, topic_num, query):
        chunk_results = self.chunks_elasticstore.max_marginal_relevance_search(query=query, k=self.chunk_k,
                                                                               filter=[{"term": {
                                                                                   "metadata.topic": topic_num}}])
        return chunk_results


class ChainRetriever(BaseRetriever):
    chains: List[BaseRetrievalQA]

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        result = []
        for chain in self.chains:
            process_result = chain(query)
            page_content = process_result["result"]
            meta_data = {}
            if "source_documents" in process_result:
                meta_data["source_documents"] = process_result["source_documents"]
            result.append(Document(page_content=page_content, metadata=meta_data))
        return result


def create_topic_qa_chain(model, vector_db, topic, **kwargs):
    chroma_embed = SentenceTransformerEmbeddings(model_name=config.FILTER_EMBEDDINGS)
    chroma_langchain = Chroma(client=vector_db, collection_name=config.ARTICLE_COLLECTION,
                              embedding_function=chroma_embed)
    chroma_search = {"k": config.ARTICLE_K, "filter": {"topic": topic}}
    chroma_retriever = chroma_langchain.as_retriever(search_kwargs=chroma_search)
    article_chain = RetrievalQA.from_chain_type(llm=model, chain_type="map_reduce", retriever=chroma_retriever,
                                                **kwargs)
    return article_chain


def topic_aggregate_chain(model, retriever, **kwargs):
    final_chain = RetrievalQA.from_chain_type(llm=model, chain_type="map_reduce",
                                              retriever=retriever, **kwargs)
    return final_chain


def create_topic_filter(model, vector_db, **kwargs):
    topic_retriever = RetrievalQA.from_chain_type(llm=model, chain_type="map_reduce",
                                                  retriever=vector_db,
                                                  **kwargs)
    format_system = SystemMessagePromptTemplate.from_template(config.TOPIC_FILTER_FORMAT_SYSTEM)
    format_user = HumanMessagePromptTemplate.from_template(config.TOPIC_FILTER_FORMAT_USER)
    format_chat = ChatPromptTemplate.from_messages([format_system, format_user])
    output_parser = CommaSeparatedListOutputParser()
    format_chat = format_chat.partial(format_instructions=output_parser.get_format_instructions())
    topic_parser = LLMChain(llm=model, prompt=format_chat, output_parser=output_parser)
    overall_chain = SimpleSequentialChain(chains=[topic_retriever, topic_parser], verbose=True)

    def filter_func(question):
        input_question = config.TOPIC_FILTER_RAW_TEMPLATE.format(question=question)
        return [int(t) for t in overall_chain(input_question)["output"]]

    return filter_func


def create_keypoints_chain(chunk_db, topic, model, k=15):
    system_prompt = SystemMessagePromptTemplate.from_template(config.TOPIC_SUM_SYSTEM)
    user_prompt = HumanMessagePromptTemplate.from_template(config.TOPIC_SUM_USER)
    overall_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
    chain_type_kwargs = {"prompt": overall_prompt}
    chain = RetrievalQA.from_chain_type(llm=model, chain_type="stuff",
                                        retriever=chunk_db.as_retriever(search_type="mmr",
                                                                        search_kwargs={'k': k, 'fetch_k': k * 3,
                                                                                       'filter': {
                                                                                           'topic': topic}}),
                                        chain_type_kwargs=chain_type_kwargs)
    return chain


async def acreate_keypoints_chains(query, topic_db, chunk_db, model, select_k=config.TOPIC_SUM_K,
                                   chunk_k=config.TOPIC_SUM_CHUNKS):
    topic_results = await afind_top_topics(topic_db, query, k=select_k)
    topic_nums = [topic.metadata["topic"] for topic in topic_results]
    return [create_keypoints_chain(chunk_db, t, model, k=chunk_k) for t in topic_nums]


async def aget_summaries(query, topic_db, chunk_db, model, select_k=config.TOPIC_SUM_K, top_k=config.TOPIC_SUM_TOP_K,
                         chunk_k=config.TOPIC_SUM_CHUNKS):
    key_chains = await acreate_keypoints_chains(query, topic_db, chunk_db, model,
                                                select_k=select_k, chunk_k=chunk_k)
    tasks = [c.acall(query) for c in key_chains]
    inter_results = asyncio.gather(*tasks)
    split_regex = re.compile(r"\n\*")

    results = []
    for r in inter_results:
        result_text = r["result"].strip()
        if result_text == "IMPOSSIBLE":
            continue
        if len(results) >= top_k:
            break
        parts = split_regex.split(result_text)

        results.append({
            "title": parts[0],
            "keypoints": parts[1:] if len(parts) > 1 else []
        })

    return results


if __name__ == "__main__":
    with open(config.ES_KEY_PATH, "r") as fp:
        es_key = fp.read().strip()
    with open(config.ES_CLOUD_ID_PATH, "r") as fp:
        es_id = fp.read().strip()

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

    query = "Which companies are developing drugs that are currently in the process of approval or just got approved?"
    agg_chain = topic_aggregate_chain(plan_llm, retriever, return_source_documents=True, verbose=True)
    test_result = agg_chain(query)
    print(test_result["result"].strip())
    print("Sources")
    for doc in test_result["source_documents"]:
        print(doc)
