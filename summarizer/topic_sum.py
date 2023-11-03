import asyncio
import datetime
import re
from joblib import Parallel, delayed
from langchain.callbacks.base import Callbacks
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.chains import RetrievalQA, LLMChain, SimpleSequentialChain
from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, \
    PromptTemplate
from langchain.schema import Document, BaseRetriever
from langchain.vectorstores import ElasticsearchStore
from typing import Any, List, Optional, Dict

try:
    import config
except ModuleNotFoundError:
    import summarizer.config as config


async def afind_top_topics(vector_db, query, now, delta, model, k=5):
    start_date = now - delta
    search_args = [{"term": {"metadata.model": model}},
                   {"range": {"metadata.recency": {"gte": start_date.strftime("%Y-%m-%d")}}}]
    topic_results = await vector_db.asimilarity_search(query=query, k=k, filter=search_args)
    return topic_results


def find_top_topics(vector_db, query, now, delta, model, k=5):
    start_date = now - delta
    search_args = [{"term": {"metadata.model": model}},
                   {"range": {"metadata.recency": {"gte": start_date.strftime("%Y-%m-%d")}}}]
    return vector_db.asimilarity_search(query, k=k, filter=search_args)


class ElasticSearchTopicRetriever(BaseRetriever):
    chunks_elasticstore: ElasticsearchStore
    chunk_k: int = config.ARTICLE_K
    topics: List[int]
    time_delta: datetime.timedelta
    now: datetime.datetime
    model: int

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        parallel = Parallel(n_jobs=len(self.topics), backend="threading", return_as="generator")
        result = []
        for chunk in parallel(delayed(self._get_relevant_chunks)(t, query) for t in self.topics):
            result.extend(chunk)
        return result

    async def aget_relevant_documents(self, query: str, *, callbacks: Callbacks = None, tags: Optional[
        List[str]] = None, metadata: Optional[Dict[str, Any]] = None, run_name: Optional[str] = None, **kwargs: Any):
        coros = []
        start_date = self.now - self.time_delta
        for t in self.topics:
            search_args = [{"term": {"metadata.topic": t}}, {"term": {"metadata.model": self.model}},
                           {"range": {"metadata.published_at": {"gte": start_date.strftime("%Y-%m-%d")}}}]
            coro = self.chunks_elasticstore.amax_marginal_relevance_search(query=query, k=self.chunk_k,
                                                                           filter=search_args)
            coros.append(coro)
        result = []
        for r in await asyncio.gather(*coros):
            result.extend(r)
        return result

    def _get_relevant_chunks(self, topic_num, query):
        start_date = self.now - self.time_delta
        search_args = [{"term": {"metadata.topic": topic_num}}, {"term": {"metadata.model": self.model}},
                       {"range": {"metadata.published_at": {"gte": start_date.strftime("%Y-%m-%d")}}}]
        chunk_results = self.chunks_elasticstore.max_marginal_relevance_search(query=query, k=self.chunk_k,
                                                                               filter=search_args)
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


def topic_aggregate_chain(model, retriever, **kwargs):
    chain_type_kwargs = {"prompt": PromptTemplate.from_template(config.QA_RESP_PROMPT)}
    final_chain = RetrievalQA.from_chain_type(llm=model, chain_type="stuff",
                                              retriever=retriever, chain_type_kwargs=chain_type_kwargs, **kwargs)
    return final_chain


def create_keypoints_chain(chunk_db, topic, topic_model, model,
                           now: datetime.datetime, delta: datetime.timedelta, k=15):
    chain_type_kwargs = {"prompt": PromptTemplate.from_template(config.TOPIC_SUM_PROMPT)}
    start_date = now - delta
    search_args = {'k': k, 'fetch_k': k * 3,
                   "filter":
                       [{"term": {"metadata.topic": topic}}, {"term": {"metadata.model": topic_model}},
                        {"range": {"metadata.published_at": {"gte": start_date.strftime("%Y-%m-%d")}}}]}
    chain = RetrievalQA.from_chain_type(llm=model, chain_type="stuff",
                                        retriever=chunk_db.as_retriever(search_type="similarity",
                                                                        search_kwargs=search_args),
                                        chain_type_kwargs=chain_type_kwargs)
    return chain


async def aget_summaries(query, topics, now, delta, topic_model, chunk_db, model, top_k=config.TOPIC_SUM_TOP_K,
                         chunk_k=config.TOPIC_SUM_CHUNKS):
    key_chains = [create_keypoints_chain(chunk_db, t, topic_model, model, now, delta, k=chunk_k) for t in topics]
    tasks = [c.acall(query) for c in key_chains]
    inter_results = await asyncio.gather(*tasks)
    split_regex = re.compile(r"\n\*")

    results = []
    for r in inter_results:
        result_text = r["result"].strip()
        if result_text == "IMPOSSIBLE":
            continue
        parts = split_regex.split(result_text)

        results.append({
            "title": parts[0],
            "keypoints": parts[1:] if len(parts) > 1 else []
        })
        if len(results) >= top_k:
            break

    return results
