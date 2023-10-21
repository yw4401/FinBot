from abc import ABC
from typing import Any, List

import chromadb
from chromadb import API, EmbeddingFunction
from chromadb.api.models.Collection import Collection
from chromadb.utils import embedding_functions
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.callbacks.base import Callbacks
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.chains import RetrievalQA, LLMChain, SimpleSequentialChain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from langchain.chat_models import ChatVertexAI
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatMessagePromptTemplate, \
    ChatPromptTemplate
from langchain.schema import Document, BaseRetriever
from langchain.vectorstores import Chroma

try:
    import config
except ModuleNotFoundError:
    import summarizer.config as config


class TopicRetriever(BaseRetriever):
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
    chroma = chromadb.PersistentClient(path="topics/topic_indices/topic-2023-4")
    chroma_articles = chromadb.PersistentClient(path="topics/topic_indices/articles-2023-4")
    chroma_embed = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=config.TOPIC_EMBEDDING)
    topic_retriever = TopicRetriever(topic_client=chroma, article_client=chroma_articles,
                                     topic_collection=config.TOPIC_COLLECTION,
                                     doc_collection=config.ARTICLE_COLLECTION,
                                     embedding=chroma_embed)

    plan_llm = ChatVertexAI(
        temperature=0,
        model_name="chat-bison",
        max_output_tokens=512
    )

    query = "Which companies are developing drugs that are currently in the process of approval?"
    agg_chain = topic_aggregate_chain(plan_llm, topic_retriever, return_source_documents=True, verbose=True)
    test_result = agg_chain(query)
    print(test_result["result"].strip())
    print("Sources")
    for doc in test_result["source_documents"]:
        print(doc)
