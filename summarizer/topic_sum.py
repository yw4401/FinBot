from abc import ABC
from typing import Any, List

import chromadb
from chromadb import API
from chromadb.api.models.Collection import Collection
from chromadb.utils import embedding_functions
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
    client: API
    collection: str
    embedding: str
    k: int = config.TOPIC_K

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        chroma_embed = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=self.embedding)
        topic_col = self.client.get_or_create_collection(name=self.collection,
                                                         embedding_function=chroma_embed)
        query_results = topic_col.query(query_texts=[query], n_results=self.k)
        results = []
        content_template = "Topic {topic}:\n{doc}"
        for topic, doc in zip(query_results["ids"][0], query_results["documents"][0]):
            results.append(Document(page_content=content_template.format(topic=topic, doc=doc)))
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


def topic_aggregate_chain(model, vector_db, topics, sub_kwargs=None, **kwargs):
    if sub_kwargs is None:
        sub_kwargs = {}
    article_chains = [create_topic_qa_chain(model, vector_db, i,
                                            return_source_documents=True, **sub_kwargs) for i in topics]
    article_subretriever = ChainRetriever(chains=article_chains)
    final_chain = RetrievalQA.from_chain_type(llm=model, chain_type="map_reduce",
                                              retriever=article_subretriever, **kwargs)
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
    topic_retriever = TopicRetriever(client=chroma, collection=config.TOPIC_COLLECTION,
                                     embedding=config.TOPIC_EMBEDDING)

    plan_llm = ChatVertexAI(
        temperature=0,
        model_name="chat-bison",
        max_output_tokens=512,
        verbose=True
    )
    query = "Which companies are developing drugs that are currently in the process of approval?"
    filter_chain = create_topic_filter(plan_llm, topic_retriever, verbose=True)
    topics = filter_chain(query)
    agg_chain = topic_aggregate_chain(plan_llm, chroma_articles, topics, return_source_documents=True)
    print(agg_chain(query))
