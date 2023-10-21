import datetime
import shutil
from pathlib import Path

import chromadb
import pandas as pd
from chromadb.utils import embedding_functions
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from tqdm import tqdm
import spacy

import config
from common import upload_blob
import torch

tqdm.pandas()
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def strip_reuter_intro(text):
    text_chunks = text.split(" - ")
    return " - ".join(text_chunks[1:]).strip()


def create_extractor(ner_recog):

    def extractor(chunks):
        ner_results = ner_recog.pipe(chunks)
        entities = []
        for doc in ner_results:
            chunk_entities = []
            for ent in doc.ents:
                chunk_entities.append(ent.text)
                entities.append(chunk_entities)
        return entities

    return extractor


def get_doc_meta(row, chunk_idx):
    published_time = datetime.datetime.fromisoformat(row["published"])
    epoch_time = datetime.datetime(1970, 1, 1, tzinfo=published_time.tzinfo)
    return {
        "title": row["title"],
        "published": (published_time - epoch_time).days,
        "doc_id": row["id"],
        "topic": row["topic"],
        "topic_prob": row["probability"],
        "source": row["source"],
        "entity": " ".join(row["entities"][chunk_idx])
    }


def build_articles_index(year, month):
    src_url = "gs://{bucket}/{file}".format(bucket=config.TOPIC_SUBSAMPLE_TARGET,
                                            file=config.TOPIC_SUBSAMPLE_FILE)
    splitter = SentenceTransformersTokenTextSplitter(model_name=config.ARTICLE_FAISS_EMBEDDING, chunk_overlap=0)
    article_df = pd.read_parquet(src_url.format(year=year, month=month))
    article_df["body"] = article_df.apply(
        lambda row: strip_reuter_intro(row["body"] if row["source"] == "reuters" else row["body"]), axis=1)
    article_df["chunks"] = article_df.body.progress_apply(splitter.split_text)
    ner_recog = spacy.load(config.NER_SPACY_MOD, enable=["ner"])
    extractor = create_extractor(ner_recog)
    article_df["entities"] = article_df.chunks.progress_apply(extractor)

    chroma_client = chromadb.PersistentClient(path=str(Path(config.ARTICLE_FAISS_TEMP_DIRECTORY, "articles")))
    chroma_embed = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=config.ARTICLE_FAISS_EMBEDDING,
                                                                            device=device)
    article_col = chroma_client.get_or_create_collection(name=config.ARTICLE_DB_COLLECTION,
                                                         embedding_function=chroma_embed)

    documents = []
    meta_datas = []
    ids = []
    cur_id = 0

    for i, row in article_df.iterrows():
        for chunk_idx, c in enumerate(row["chunks"]):
            documents.append(c)
            meta_datas.append(get_doc_meta(row, chunk_idx))
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
