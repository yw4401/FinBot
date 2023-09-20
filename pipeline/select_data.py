import pandas as pd
from google.cloud import storage
from datetime import datetime
import config
from pandarallel import pandarallel
from contextlib import closing
import json


pandarallel.initialize(progress_bar=True)


def get_work(source=config.ARTICLE_COREF_TARGET_BUCKET, filter_f=lambda article: True):
    with closing(storage.Client(project=config.GCP_PROJECT)) as client:
        tbd = set([f.name for f in client.list_blobs(bucket_or_name=source)])
        tbd_df = pd.DataFrame({
            "tbd": list(tbd)
        })

    def filter_article(f_name):
        with closing(storage.Client(project=config.GCP_PROJECT)) as client:
            bucket = client.bucket(source)
            with bucket.blob(f_name).open("r") as fp:
                article = json.load(fp)
                if filter_f(article):
                    return article
        return ""

    tbd_df["result"] = tbd_df.tbd.parallel_apply(filter_article)
    tbd_df_filtered = tbd_df.loc[tbd_df["result"].str.len() > 0]

    return tbd_df_filtered.result.to_list()


def year_month_filter(article, year, month):
    timestamp = datetime.fromisoformat(article["published"])
    return timestamp.year == year and timestamp.month == month


if __name__ == "__main__":
    year = 2023
    month = 4
    sample = get_work(filter_f=lambda a: year_month_filter(article=a, year=2023, month=4))
    for s in sample:
        if s["source"] == "reuters" and len(s["title"]) < len(s["category"]):
            real_title = s["category"]
            s["category"] = s["title"]
            s["title"] = real_title

    final_df = pd.DataFrame(sample)

    target_url = "gs://{bucket}/{file}".format(bucket=config.ARTICLE_COREF_SUBSAMPLE_TARGET,
                                               file=config.ARTICLE_COREF_SUBSAMPLE_FILE)
    final_df.to_parquet(target_url.format(year=year, month=month), index=False)
