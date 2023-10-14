import datetime
import shutil
import os
from pathlib import Path

import chromadb
import numpy as np
import pandas as pd
from chromadb.utils import embedding_functions
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import config
from common import upload_blob

tqdm.pandas()


def get_doc_meta(row):
    published_time = datetime.datetime.fromisoformat(row["published"])
    epoch_time = datetime.datetime(1970, 1, 1, tzinfo=published_time.tzinfo)
    return {
        "title": row["title"],
        "published": (published_time - epoch_time).days,
        "doc_id": row["id"],
        "topic": row["topic"],
        "topic_prob": row["probability"],
        "source": row["source"]
    }


def build_articles_index(year, month):
    src_url = "gs://{bucket}/{file}".format(bucket=config.TOPIC_SUBSAMPLE_TARGET,
                                            file=config.TOPIC_SUBSAMPLE_FILE)
    splitter = SentenceTransformersTokenTextSplitter(model_name=config.ARTICLE_FAISS_EMBEDDING, chunk_overlap=0)
    article_df = pd.read_parquet(src_url.format(year=year, month=month))
    article_df["chunks"] = article_df.body.progress_apply(splitter.split_text)

    chroma_client = chromadb.PersistentClient(path=str(Path(config.ARTICLE_FAISS_TEMP_DIRECTORY, "articles")))
    chroma_embed = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=config.ARTICLE_FAISS_EMBEDDING,
                                                                            device="cuda")
    article_col = chroma_client.get_or_create_collection(name=config.ARTICLE_DB_COLLECTION,
                                                         embedding_function=chroma_embed)

    documents = []
    meta_datas = []
    ids = []
    cur_id = 0

    for i, row in article_df.iterrows():
        for c in row["chunks"]:
            documents.append(c)
            meta_datas.append(get_doc_meta(row))
            ids.append(str(cur_id))
            cur_id += 1

    article_col.add(ids=ids, documents=documents, metadatas=meta_datas)

    final_dir = Path(config.ARTICLE_FAISS_TEMP_DIRECTORY, "articles")
    final_file = Path(config.ARTICLE_FAISS_TEMP_DIRECTORY, "target")
    dest_file = config.ARTICLE_FAISS_FILE.format(year=year, month=month)
    shutil.make_archive(final_file, 'zip', final_dir)
    upload_blob(bucket_name=config.ARTICLE_FAISS_TARGET,
                source_file_name=Path(config.ARTICLE_FAISS_TEMP_DIRECTORY, "target.zip"),
                destination_blob_name=dest_file, generation=None)


def build_topic_index(year, month):
    src_url = "gs://{bucket}/{file}".format(bucket=config.TOPIC_SUM_TARGET,
                                            file=config.TOPIC_SUM_TARGET_FILE)
    topic_sum_df = pd.read_parquet(src_url.format(year=year, month=month))

    chroma_client = chromadb.PersistentClient(path=str(Path(config.ARTICLE_FAISS_TEMP_DIRECTORY, "topics")))
    chroma_embed = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=config.ARTICLE_FAISS_EMBEDDING,
                                                                            device="cuda")
    topic_col = chroma_client.get_or_create_collection(name=config.TOPICS_DB_COLLECTION,
                                                       embedding_function=chroma_embed)

    documents = []
    ids = []

    for r, row in topic_sum_df.iterrows():
        documents.append(row["summary"])
        ids.append(str(row["topics"]))

    topic_col.add(ids=ids, documents=documents)

    final_file = Path(config.ARTICLE_FAISS_TEMP_DIRECTORY, "topics_target")
    final_dir = Path(config.ARTICLE_FAISS_TEMP_DIRECTORY, "topics")
    shutil.make_archive(final_file, 'zip', final_dir)
    dest_file = config.TOPIC_FAISS_FILE.format(year=year, month=month)
    upload_blob(bucket_name=config.TOPIC_FAISS_TARGET,
                source_file_name=Path(config.ARTICLE_FAISS_TEMP_DIRECTORY, "topics_target.zip"),
                destination_blob_name=dest_file, generation=None)


if __name__ == "__main__":
    year = 2023
    month = 4
    try:
        print("Building Article Indices")
        build_articles_index(year=year, month=month)
        print("Building Topic Indices")
        build_topic_index(year=year, month=month)
    finally:
        try:
            shutil.rmtree(config.ARTICLE_FAISS_TEMP_DIRECTORY)
        except FileNotFoundError:
            pass
