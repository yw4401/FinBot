from contextlib import closing

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
import config
import google.cloud.bigquery as bq
import google.cloud.bigquery.dbapi as bqapi

from summarize_topics_6 import summarization_wrapper, create_topic_summarizer, create_palm2_chain


def get_topicless_articles(client: bq.Client):
    query = "SELECT id, title, body " \
            "FROM Articles.CleanedArticles WHERE topic IS NULL"
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
    summarizer = create_topic_summarizer("lc", chain=create_palm2_chain(credentials=None),
                                         adapter=lambda x: x["text"]["summary"].strip())
    parallel = Parallel(n_jobs=jobs, backend="threading", return_as="generator")
    topic_sum = pd.DataFrame({
        "topic": topics,
        "summary": ["" for _ in topics]
    })

    with tqdm(total=len(topics)) as progress:
        for i, (w, summary) in enumerate(
                parallel(delayed(summarization_wrapper)(summarizer, work, df) for work in topics)):
            topic_sum.loc[topic_sum.topics == w, "summary"] = summary
            progress.update(1)
            if i % 10 == 0:
                print(summary)

    return topic_sum


def write_topics(topic_df: pd.DataFrame, sum_df, project):
    upd_stmt = """UPDATE Articles.CleanedArticles SET topic = %s, topic_prob = %s WHERE id = %s"""
    ins_stmt = """INSERT INTO Articles.TopicSummary VALUES(%s, %s)"""
    client = bq.Client(project=project)

    with closing(bqapi.Connection(client=client)) as connection:
        with closing(connection.cursor()) as cursor:
            cursor.executemany(upd_stmt,
                               [(row["topic"], row["topic_prob"], row["id"]) for _, row in topic_df.iterrows()])
            cursor.executemany(ins_stmt, [(row["topic"], row["summary"]) for _, row in sum_df.iterrows()])
    return True


if __name__ == "__main__":
    client = bq.Client(project=config.GCP_PROJECT)
    articles = get_topicless_articles(client)
    articles, _ = identify_topics(articles)
    sum_df = summarize_topics(articles)
    write_topics(articles, sum_df, config.GCP_PROJECT)



