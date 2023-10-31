from contextlib import closing

import pandas as pd
import google.cloud.bigquery as bq
import google.cloud.bigquery.dbapi as bqapi
import spacy
import torch

from deduplication import execute_deduplication
import config
from coref_resolve import add_coreference_resolution
from allennlp.predictors.predictor import Predictor
from joblib import delayed, Parallel
from tqdm import tqdm


def get_scraped_articles(client: bq.Client):
    query = "SELECT SA.id, SA.url, SA.source, SA.title, SA.published, SA.body, SA.summary, SA.summary_type, SA.category " \
            "FROM Articles.ScrapedArticles AS SA LEFT JOIN Articles.CleanedArticles CA ON SA.id = CA.id " \
            "WHERE SA.published >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY) AND CA.id IS NULL"
    results = []

    with closing(bqapi.Connection(client=client)) as connection:
        with closing(connection.cursor()) as cursor:
            cursor.execute(query)
            for r in cursor.fetchall():
                results.append(list(r))
    return pd.DataFrame(results,
                        columns=["id", "url", "source", "title", "published", "body", "summary", "summary_type",
                                 "category"])


def write_batch(project, batch):
    stmt = """INSERT INTO Articles.CleanedArticles VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
    client = bq.Client(project=project, credentials=None)
    with closing(bqapi.Connection(client=client)) as connection:
        with closing(connection.cursor()) as cursor:
            cursor.executemany(stmt, batch)
        connection.commit()
    return True


def write_coref_articles(project, df, batch=10, jobs=8):
    params = [(o["id"], o["url"], o["source"], o["title"], o["published"],
               o["body"], o["coref"], o["summary"], o["summary_type"],
               o["category"]) for i, o in df.iterrows()]
    batches = []
    for i in range(0, len(params), batch):
        batches.append(params[i:i + batch])

    parallel = Parallel(n_jobs=jobs, return_as="generator")
    with tqdm(total=len(batches)) as progress:
        for _ in parallel(delayed(write_batch)(project, b) for b in batches):
            progress.update(1)


if __name__ == "__main__":
    client = bq.Client(project=config.GCP_PROJECT, credentials=None)
    predictor = Predictor.from_path(config.ARTICLE_COREF_MOD_URL, cuda_device=torch.cuda.current_device())
    nlp = spacy.load(config.ARTICLE_COREF_SPACY_MOD)

    src_df = get_scraped_articles(client)
    cleaned_df = execute_deduplication(src_df)
    coref_df = add_coreference_resolution(cleaned_df, predictor=predictor, nlp=nlp)
    write_coref_articles(config.GCP_PROJECT, coref_df, batch=100)
