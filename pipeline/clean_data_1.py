from contextlib import closing

import pandas as pd
import google.cloud.bigquery as bq
import google.cloud.bigquery.dbapi as bqapi
import spacy
import torch

from deduplication import execute_deduplication # Importing function for deduplication
import config # Importing configuration settings
from coref_resolve import add_coreference_resolution # Importing function for coreference resolution
from allennlp.predictors.predictor import Predictor # Importing AllenNLP Predictor
from joblib import delayed, Parallel # Importing joblib for parallel processing
from tqdm import tqdm # Importing tqdm for progress tracking


def get_scraped_articles(client: bq.Client):
    # SQL query to fetch scraped articles within a specific timeframe
    query = "SELECT SA.id, SA.url, SA.source, SA.title, SA.published, SA.body, SA.summary, SA.summary_type, SA.category " \
            "FROM Articles.ScrapedArticles AS SA " \
            f"WHERE SA.published >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {config.ARTICLE_INGEST_MAX_DAYS} DAY)"

    # SQL query to fetch cleaned articles within a specific timeframe
    cleaned_query = "SELECT CA.id FROM Articles.CleanedArticles AS CA " \
                    "WHERE CA.published >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), " \
                    f"INTERVAL {config.ARTICLE_INGEST_MAX_DAYS + 5} DAY)"
    results = []
    existing = set()

    with closing(bqapi.Connection(client=client)) as connection:
        with closing(connection.cursor()) as cursor:
            cursor.execute(query)
            for r in cursor.fetchall():
                results.append(list(r))
            cursor.execute(cleaned_query)
            for r in cursor.fetchall():
                existing.add(r.id)
    temp_result = pd.DataFrame(results,
                               columns=["id", "url", "source", "title", "published", "body", "summary", "summary_type",
                                        "category"])
    temp_result["exists"] = temp_result.id.apply(lambda i: i in existing)
    return temp_result


# Function to write a batch of cleaned articles to BigQuery
def write_batch(project, batch):
    stmt = """INSERT INTO Articles.CleanedArticles VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
    client = bq.Client(project=project)
    with closing(bqapi.Connection(client=client)) as connection:
        with closing(connection.cursor()) as cursor:
            cursor.executemany(stmt, batch)
        connection.commit()
    return True


# Function to write articles after coreference resolution to BigQuery in batches
def write_coref_articles(project, df, batch=10, jobs=8):
    params = [(o["id"], o["url"], o["source"], o["title"], o["published"],
               o["body"], o["coref"], o["summary"], o["summary_type"],
               o["category"]) for i, o in df.iterrows()]
    batches = []
    for i in range(0, len(params), batch):
        batches.append(params[i:i + batch])

    # Using joblib for parallel execution and tracking progress with tqdm
    parallel = Parallel(n_jobs=jobs, return_as="generator")
    with tqdm(total=len(batches)) as progress:
        for _ in parallel(delayed(write_batch)(project, b) for b in batches):
            progress.update(1)


if __name__ == "__main__":
    client = bq.Client(project=config.GCP_PROJECT)
    predictor = Predictor.from_path(config.ARTICLE_COREF_MOD_URL, cuda_device=torch.cuda.current_device())
    nlp = spacy.load(config.ARTICLE_COREF_SPACY_MOD)

    # Fetching scraped articles within a specified timeframe
    src_df = get_scraped_articles(client)
    print(f"Total Articles: {src_df.shape[0]}")

    # Executing deduplication process on scraped articles
    cleaned_df = execute_deduplication(src_df).copy()
    print(f"De-dup Articles: {cleaned_df.shape[0]}")

    # Filtering out existing articles from the cleaned set
    cleaned_df = cleaned_df.loc[~cleaned_df.exists]
    print(f"Final Articles: {cleaned_df.shape[0]}")

    # Adding coreference resolution to the final set of articles
    coref_df = add_coreference_resolution(cleaned_df, predictor=predictor, nlp=nlp)

    # Writing the articles with coreference resolution to BigQuery in batches
    write_coref_articles(config.GCP_PROJECT, coref_df, batch=100)
