from contextlib import closing

import datetime

import pandas as pd
import spacy
import torch
from elasticsearch import Elasticsearch, NotFoundError
from langchain.text_splitter import SpacyTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer
import google.cloud.bigquery as bq
import google.cloud.bigquery.dbapi as bqapi

import config

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


def create_es_topic_doc(encoder: SentenceTransformer, row):
    embedding = encoder.encode(row["summary"])
    return {
        "description": row["summary"],
        "description_embedding": embedding.tolist(),
        "metadata": {
            "topic": int(row["topic"]),
            "model": int(row["model"]),
            "recency": row["recency"],
        }
    }


def create_es_topic_idx(client: Elasticsearch, encoder, topic_sum_df):
    topic_id_format = "{topic_model}-{topic_num}"
    try:
        client.indices.get(index=config.ES_TOPIC_INDEX)
    except NotFoundError as e:
        client.indices.create(index=config.ES_TOPIC_INDEX, mappings=config.ES_TOPIC_MAPPING)
        client.indices.get(index=config.ES_TOPIC_INDEX)

    with tqdm(total=topic_sum_df.shape[0]) as progress:
        for i, topic in topic_sum_df.iterrows():
            doc_id = topic_id_format.format(topic_model=topic["model"], topic_num=topic["topic"])
            client.update(index=config.ES_TOPIC_INDEX, id=doc_id,
                          doc=create_es_topic_doc(encoder, topic), doc_as_upsert=True)
            progress.update(1)


def create_es_chunk_docs(encoder: SentenceTransformer, row):
    embeddings = encoder.encode(row["chunks"])
    published = row["published"].replace(tzinfo=datetime.timezone.utc).strftime('%Y-%m-%d')
    for i, chunk in enumerate(row["chunks"]):
        chunk_id = "{article_id}-{chunk_idx}".format(article_id=row["id"], chunk_idx=i)
        yield chunk_id, {
            "chunk_text": chunk,
            "chunk_text_embedding": embeddings[i].tolist(),
            "metadata": {
                "entities": " ".join(row["entities"][i]),
                "published_at": published,
                "topic": int(row["topic"]),
                "model": int(row["model"])
            }
        }


def create_es_doc_idx(client: Elasticsearch, encoder, article_df):
    try:
        client.indices.get(index=config.ES_ARTICLE_INDEX)
    except NotFoundError:
        client.indices.create(index=config.ES_ARTICLE_INDEX, mappings=config.ES_ARTICLES_MAPPING)
        client.indices.get(index=config.ES_ARTICLE_INDEX)

    article_df = preprocess_articles(article_df, splitter=create_splitter())

    with tqdm(total=article_df.shape[0]) as progress:
        for i, row in article_df.iterrows():
            for id_chunk, doc in create_es_chunk_docs(encoder, row):
                client.update(index=config.ES_ARTICLE_INDEX, id=id_chunk, doc=doc, doc_as_upsert=True)
            progress.update(1)


def get_unindexed_topics(client: bq.Client):
    query = "SELECT TS.model, TS.topic, TDT.recency, TS.summary FROM "\
                f"(SELECT TAT.model as model, TAT.topic as topic, " \
                f"timestamp_seconds(cast(avg(unix_seconds(CA.published)) as int64)) AS recency " \
                f"FROM Articles.ArticleTopic AS TAT, Articles.CleanedArticles AS CA " \
                f"WHERE TAT.article_id = CA.id AND TAT.topic_prob >= {config.TOPIC_EMBED_TOP_THRESHOLD} " \
                f"GROUP BY TAT.model, TAT.topic) AS TDT, " \
            f"Articles.TopicSummary AS TS, " \
                f"(SELECT TM.id AS id, MAX(fit_date) FROM Articles.TopicModel as TM " \
                f"WHERE NOT TM.servable GROUP BY TM.id) AS NM " \
            f"WHERE NM.id = TDT.model AND TDT.model = TS.model AND TDT.topic = TS.topic ORDER BY TS.topic ASC"
    result = []
    with closing(bqapi.Connection(client=client)) as connection:
        with closing(connection.cursor()) as cursor:
            cursor.execute(query)
            for r in cursor.fetchall():
                result.append(list(r))

    return pd.DataFrame(result, columns=["model", "topic", "recency", "summary"])


def get_articles_by_topics(client: bq.Client, model):
    query = "SELECT CA.id, CA.published, CA.source, CA.body, ACT.topic FROM Articles.CleanedArticles AS CA, " \
            "Articles.ArticleTopic ACT " \
            "WHERE CA.id = ACT.article_id AND ACT.model = %s"
    result = []
    with closing(bqapi.Connection(client=client)) as connection:
        with closing(connection.cursor()) as cursor:
            cursor.execute(query, (int(model), ))
            for r in cursor.fetchall():
                result.append(list(r))
    result_df = pd.DataFrame(result, columns=["id", "published", "source", "body", "topic"])
    result_df["model"] = model
    return result_df


if __name__ == "__main__":
    with open(config.ES_KEY_PATH, "r") as fp:
        es_key = fp.read().strip()
    with open(config.ES_CLOUD_ID_PATH, "r") as fp:
        es_id = fp.read().strip()
    print(device)
    elastic_client = Elasticsearch(cloud_id=es_id, api_key=es_key)
    encoder = SentenceTransformer(model_name_or_path=config.TOPIC_FAISS_EMBEDDING, device=device)

    with closing(bq.Client(project=config.GCP_PROJECT)) as client:
        topic_df = get_unindexed_topics(client=client)
        article_df = get_articles_by_topics(client=client, model=topic_df.model.iloc[0])
        print("Building Topic Indices")
        create_es_topic_idx(client=elastic_client, encoder=encoder, topic_sum_df=topic_df)
        print("Building Article Indices")
        create_es_doc_idx(client=elastic_client, encoder=encoder, article_df=article_df)
