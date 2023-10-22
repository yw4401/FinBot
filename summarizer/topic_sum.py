from typing import Any, List

from chromadb import API
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


class ChromaTopicRetriever(BaseRetriever):
    topic_client: API
    article_client: API
    topic_collection: str
    doc_collection: str
    embedding: Any
    topic_k: int = config.TOPIC_K
    chunk_k: int = config.ARTICLE_K

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        topic_col = self.topic_client.get_or_create_collection(name=self.topic_collection,
                                                               embedding_function=self.embedding)
        doc_col = self.article_client.get_or_create_collection(name=self.doc_collection,
                                                               embedding_function=self.embedding)
        query_results = topic_col.query(query_texts=[query], n_results=self.topic_k)
        results = []
        for topic, _ in zip(query_results["ids"][0], query_results["documents"][0]):
            topic_qresults = doc_col.query(query_texts=query, where={"topic": int(topic)}, n_results=self.chunk_k)
            for chunk, meta in zip(topic_qresults["documents"][0], topic_qresults["metadatas"][0]):
                results.append(Document(page_content=chunk, metadata=meta))
        return results


class ElasticSearchTopicRetriever(BaseRetriever):
    topic_elasticstore: ElasticsearchStore
    chunks_elasticstore: ElasticsearchStore
    topic_k: int = config.TOPIC_K
    chunk_k: int = config.ARTICLE_K

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        topic_results = self.topic_elasticstore.max_marginal_relevance_search(query, k=self.topic_k)
        parallel = Parallel(n_jobs=self.topic_k, backend="threading", return_as="generator")
        result = []
        topic_nums = [topic.metadata["topic"] for topic in topic_results]
        for chunk in parallel(delayed(self._get_relevant_chunks)(t, query) for t in topic_nums):
            result.extend(chunk)
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
    chroma_embed = SentenceTransformerEmbeddings(model_name=config.TOPIC_EMBEDDING)
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
        temperature=0,
        model_name="chat-bison",
        max_output_tokens=512
    )

    query = "Which companies are developing drugs that are currently in the process of approval?"
    agg_chain = topic_aggregate_chain(plan_llm, retriever, return_source_documents=True, verbose=True)
    test_result = agg_chain(query)
    print(test_result["result"].strip())
    print("Sources")
    for doc in test_result["source_documents"]:
        print(doc)
