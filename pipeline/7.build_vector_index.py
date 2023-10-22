import datetime
import shutil
from pathlib import Path

import chromadb
import pandas as pd
import spacy
import torch
from chromadb.utils import embedding_functions
from elasticsearch import Elasticsearch, NotFoundError
from langchain.text_splitter import SpacyTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer

import config
from common import upload_blob

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
                if ent.label_ in config.ES_ARTICLE_ENTITIES:
                    chunk_entities.append(ent.text)
            entities.append(chunk_entities)
        return entities

    return extractor


def get_doc_chroma_meta(row, chunk_idx):
    published_time = datetime.datetime.fromisoformat(row["published"])
    epoch_time = datetime.datetime(1970, 1, 1, tzinfo=published_time.tzinfo)
    return {
        "title": row["title"],
        # "url": row["url"],
        "published": (published_time - epoch_time).days,
        "doc_id": row["id"],
        "topic": row["topic"],
        "topic_prob": row["probability"],
        "source": row["source"],
        "entity": " ".join(row["entities"][chunk_idx])
    }


def create_splitter():
    tokenizer = AutoTokenizer.from_pretrained(config.ARTICLE_SPLITTER_TOKENIZER)

    def _huggingface_tokenizer_length(text: str) -> int:
        return len(tokenizer.encode(text))

    splitter = SpacyTextSplitter(length_function=_huggingface_tokenizer_length,
                                 chunk_overlap=config.ARTICLE_SPLITTER_CHUNK_OVERLAP,
                                 chunk_size=config.ARTICLE_SPLITTER_CHUNK_SIZE)
    return splitter


def preprocess_articles(article_df, splitter):
    article_df = article_df.copy()
    article_df["body"] = article_df.apply(
        lambda row: strip_reuter_intro(row["body"] if row["source"] == "reuters" else row["body"]), axis=1)
    article_df["chunks"] = article_df.body.progress_apply(splitter.split_text)
    ner_recog = spacy.load(config.NER_SPACY_MOD, enable=["ner"])
    extractor = create_extractor(ner_recog)
    article_df["entities"] = article_df.chunks.progress_apply(extractor)
    return article_df


def build_chroma_articles_index(year, month):
    src_url = "gs://{bucket}/{file}".format(bucket=config.TOPIC_SUBSAMPLE_TARGET,
                                            file=config.TOPIC_SUBSAMPLE_FILE)
    splitter = create_splitter()
    article_df = pd.read_parquet(src_url.format(year=year, month=month))
    article_df = preprocess_articles(article_df, splitter)

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
            meta_datas.append(get_doc_chroma_meta(row, chunk_idx))
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


def build_chroma_topic_index(year, month):
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


def create_es_topic_doc(encoder: SentenceTransformer, row):
    embedding = encoder.encode(row["summary"])
    return {
        "description": row["summary"],
        "description_embedding": embedding.tolist(),
        "metadata": {
            "topic": int(row["topics"]),
            "created_at": row["created_at"],
        }
    }


def create_es_topic_idx(client: Elasticsearch, encoder, topic_sum_df):
    topic_id_format = "{topic_created_at}-{topic_num}"
    try:
        client.indices.get(index=config.ES_TOPIC_INDEX)
    except NotFoundError as e:
        client.indices.create(index=config.ES_TOPIC_INDEX, mappings=config.ES_TOPIC_MAPPING)
        client.indices.get(index=config.ES_TOPIC_INDEX)

    with tqdm(total=topic_sum_df.shape[0]) as progress:
        for i, topic in topic_sum_df.iterrows():
            doc_id = topic_id_format.format(topic_created_at=topic["created_at"], topic_num=topic["topics"])
            client.update(index=config.ES_TOPIC_INDEX, id=doc_id,
                          doc=create_es_topic_doc(encoder, topic), doc_as_upsert=True)
            progress.update(1)


def create_es_chunk_docs(encoder: SentenceTransformer, row):
    embeddings = encoder.encode(row["chunks"])
    published = datetime.datetime.fromisoformat(row["published"]).replace(tzinfo=datetime.timezone.utc)
    published = published.strftime('%Y-%m-%d')
    for i, chunk in enumerate(row["chunks"]):
        chunk_id = "{article_id}-{chunk_idx}".format(article_id=row["id"], chunk_idx=i)
        yield chunk_id, {
            "chunk_text": chunk,
            "chunk_text_embedding": embeddings[i].tolist(),
            "metadata": {
                "entities": " ".join(row["entities"][i]),
                "published_at": published,
                "topic": int(row["topic"])
            }
        }


def create_es_doc_idx(client: Elasticsearch, encoder, article_df):
    try:
        client.indices.get(index=config.ES_ARTICLE_INDEX)
    except NotFoundError as e:
        client.indices.create(index=config.ES_ARTICLE_INDEX, mappings=config.ES_ARTICLES_MAPPING)
        client.indices.get(index=config.ES_ARTICLE_INDEX)

    article_df = preprocess_articles(article_df, splitter=create_splitter())

    with tqdm(total=article_df.shape[0]) as progress:
        for i, row in article_df.iterrows():
            for id_chunk, doc in create_es_chunk_docs(encoder, row):
                client.update(index=config.ES_ARTICLE_INDEX, id=id_chunk, doc=doc, doc_as_upsert=True)
            progress.update(1)


if __name__ == "__main__":
    year = 2023
    month = 4

    with open(config.ES_KEY_PATH, "r") as fp:
        es_key = fp.read().strip()
    with open(config.ES_CLOUD_ID_PATH, "r") as fp:
        es_id = fp.read().strip()
    elastic_client = Elasticsearch(cloud_id=es_id, api_key=es_key)
    encoder = SentenceTransformer(model_name_or_path=config.TOPIC_FAISS_EMBEDDING, device=device)
    topic_url = "gs://{bucket}/{file}".format(bucket=config.TOPIC_SUM_TARGET,
                                              file=config.TOPIC_SUM_TARGET_FILE)
    topic_sum_df = pd.read_parquet(topic_url.format(year=year, month=month))
    topic_sum_df["created_at"] = f"{year}-{month:02d}-30"
    article_url = "gs://{bucket}/{file}".format(bucket=config.TOPIC_SUBSAMPLE_TARGET,
                                                file=config.TOPIC_SUBSAMPLE_FILE)
    article_df = pd.read_parquet(article_url.format(year=year, month=month))

    print("Building Topic Indices")
    create_es_topic_idx(client=elastic_client, encoder=encoder, topic_sum_df=topic_sum_df)
    print("Building Article Indices")
    create_es_doc_idx(client=elastic_client, encoder=encoder, article_df=article_df)
