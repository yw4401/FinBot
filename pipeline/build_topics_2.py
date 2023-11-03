import os
from contextlib import closing

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from google.oauth2 import service_account
from bertopic.vectorizers import ClassTfidfTransformer
import config
import google.cloud.bigquery as bq
import google.cloud.bigquery.dbapi as bqapi

from pipeline.common import download_blob, upload_blob, BigquerySession
from summarize_topics import summarization_wrapper, create_topic_summarizer, create_palm2_chain
import shutil
import uuid
import gc


def get_topicless_articles(client, model):
    query = "SELECT CA.id, CA.title, CA.body, CA.published " \
            "FROM Articles.CleanedArticles AS CA LEFT JOIN Articles.ArticleTopic TA ON CA.id = TA.article_id " \
            f"WHERE TA.article_id IS NULL AND (TA.model != %s OR TA.model IS NULL) AND " \
            f"CA.published >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {config.TOPIC_FIT_RANGE_DAY} DAY)"
    results = []

    with closing(bqapi.Connection(client=client)) as connection:
        with closing(connection.cursor()) as cursor:
            cursor.execute(query, (model,))
            for r in cursor.fetchall():
                results.append(list(r))
    return pd.DataFrame(results,
                        columns=["id", "title", "body", "published"])


def get_fitting_articles(client):
    query = "SELECT id, title, body, published " \
            "FROM Articles.CleanedArticles AS CA WHERE CA.published >= " \
            f"TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {config.TOPIC_FIT_RANGE_DAY} DAY)"
    results = []
    with closing(bqapi.Connection(client=client)) as connection:
        with closing(connection.cursor()) as cursor:
            cursor.execute(query)
            for r in cursor.fetchall():
                results.append(list(r))
        return pd.DataFrame(results,
                            columns=["id", "title", "body", "published"])


def identify_topics(df: pd.DataFrame):
    embedding_model = SentenceTransformer(config.TOPIC_EMBEDDING)

    umap_model = UMAP(n_neighbors=config.TOPIC_UMAP_NEIGHBORS, n_components=config.TOPIC_UMAP_COMPONENTS,
                      min_dist=config.TOPIC_UMAP_MIN_DIST, metric=config.TOPIC_UMAP_METRIC)

    hdbscan_model = HDBSCAN(min_cluster_size=config.TOPIC_HDBSCAN_MIN_SIZE,
                            metric=config.TOPIC_HDBSCAN_METRIC,
                            cluster_selection_method='eom',
                            prediction_data=True)

    vectorizer_model = CountVectorizer(stop_words="english")

    ctfidf_model = ClassTfidfTransformer()

    # All steps together
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
    )
    topics, prob = topic_model.fit_transform(df.body)
    df["topic"] = topics
    df["topic_prob"] = prob
    return df, topic_model


def summarize_topics(df: pd.DataFrame, jobs=7):
    topics = list(set(df.topic.unique()))
    credentials = service_account.Credentials.from_service_account_file(config.VERTEX_AI_KEY_PATH)
    summarizer = create_topic_summarizer("lc", chain=create_palm2_chain(credentials=credentials),
                                         adapter=lambda x: x["text"]["summary"].strip())
    parallel = Parallel(n_jobs=jobs, backend="threading", return_as="generator")
    topic_sum = pd.DataFrame({
        "topic": topics,
        "summary": ["" for _ in topics]
    })

    with tqdm(total=len(topics)) as progress:
        for i, (w, summary) in enumerate(
                parallel(delayed(summarization_wrapper)(summarizer, work, df) for work in topics)):
            topic_sum.loc[topic_sum.topic == w, "summary"] = summary
            progress.update(1)
            if i % 10 == 0:
                print(summary)

    return topic_sum


def upload_topic(model: BERTopic):
    path_name = str(uuid.uuid4())
    model.save(path="./temp_model", serialization="safetensors", save_embedding_model=False, save_ctfidf=True)
    shutil.make_archive(path_name, "zip", "./temp_model")
    upload_blob(config.TOPIC_BUCKET, path_name + ".zip", destination_blob_name=path_name + ".zip", generation=None)
    shutil.rmtree("./temp_model")
    os.remove(path_name + ".zip")
    return f"gs://{config.TOPIC_BUCKET}/{path_name}.zip"


def write_topics(client: bq.Client, model, sum_df):
    path = upload_topic(model)
    model_stmt = """INSERT INTO Articles.TopicModel 
    SELECT CASE WHEN MAX(id) IS NULL THEN 0 ELSE MAX(id) + 1 END, CURRENT_TIMESTAMP(), ?, False 
    FROM Articles.TopicModel;"""
    ins_stmt = """INSERT INTO Articles.TopicSummary SELECT ?, MAX(id), ? FROM Articles.TopicModel"""
    id_stmt = """SELECT MAX(id) FROM Articles.TopicModel"""
    client = bq.Client(project=client.project)

    with BigquerySession(client) as session:
        session.begin_transaction()
        job = client.query(model_stmt, bq.QueryJobConfig(
            create_session=False,
            query_parameters=[
                bq.ScalarQueryParameter(None, "STRING", path)
            ],
            connection_properties=[
                bq.query.ConnectionProperty(
                    key="session_id", value=session.session_id
                )
            ],
        ), location=session.location)
        job.result()
        for _, row in sum_df.iterrows():
            job = client.query(ins_stmt, bq.QueryJobConfig(
                create_session=False,
                query_parameters=[
                    bq.ScalarQueryParameter(None, "INTEGER", row["topic"]),
                    bq.ScalarQueryParameter(None, "STRING", row["summary"])
                ],
                connection_properties=[
                    bq.query.ConnectionProperty(
                        key="session_id", value=session.session_id
                    )
                ],
            ), location=session.location)
            job.result()
        session.commit()


def load_topic_model(client):
    query = "SELECT id, fit_date, path FROM Articles.TopicModel AS TM " \
            f"WHERE TM.fit_date >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {config.TOPIC_TTL_DAY} DAY) ORDER BY fit_date DESC LIMIT 1"
    with closing(bqapi.Connection(client=client)) as connection:
        with closing(connection.cursor()) as cursor:
            cursor.execute(query)
            result = cursor.fetchone()
            if result is None:
                raise FileNotFoundError("Model Unavailable")

    path = result[2]
    id = result[0]
    download_blob(path, "./temp_topic.zip")
    shutil.unpack_archive("./temp_topic.zip", "./temp_topic")
    model = BERTopic.load("./temp_topic", embedding_model=SentenceTransformer(config.TOPIC_EMBEDDING))
    shutil.rmtree('./temp_topic')
    os.remove("./temp_topic.zip")
    return id, model


def get_topic_model(client):
    try:
        return load_topic_model(client)
    except FileNotFoundError:
        articles = get_fitting_articles(client)
        articles, model = identify_topics(articles)
        sum_df = summarize_topics(articles)
        write_topics(client, model, sum_df)
        return load_topic_model(client)


def batch_insert_topic(project, batch):
    query = """INSERT INTO Articles.ArticleTopic VALUES(%s, %s, %s, %s)"""
    with closing(bq.Client(project=project)) as client:
        with closing(bqapi.Connection(client=client)) as connection:
            with closing(connection.cursor()) as cursor:
                cursor.executemany(query, batch)
            connection.commit()


def categorize_articles(client):
    id, model = get_topic_model(client)
    articles = get_topicless_articles(client, id)
    if articles.shape[0] == 0:
        return
    topics, prob = model.transform(articles.body)
    articles["topic"] = topics
    articles["topic_prob"] = prob
    return id, articles


def write_article_topics(articles, model_id, batch_size=100, jobs=8):
    params = [(row["id"], model_id, row["topic"], row["topic_prob"]) for _, row in articles.iterrows()]
    batches = []
    for i in range(0, len(params), batch_size):
        batches.append(params[i:i + batch_size])
    parallel = Parallel(n_jobs=jobs, return_as="generator")
    with tqdm(total=len(batches)) as progress:
        for _ in parallel(delayed(batch_insert_topic)(client.project, b) for b in batches):
            progress.update(1)


if __name__ == "__main__":
    with closing(bq.Client(project=config.GCP_PROJECT, credentials=None)) as client:
        mid, articles = categorize_articles(client)
        gc.collect()
        write_article_topics(articles, mid)
