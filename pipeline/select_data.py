import pandas as pd
from google.cloud import storage
from datetime import datetime
import config
from contextlib import closing
from multiprocessing import cpu_count
import json
import os
import multiprocessing as mp
import numpy as np
from tqdm import tqdm


class FilterArticle:

    def __init__(self, filter_f, source):
        self.filter_f = filter_f
        self.source = source

    def __call__(self, pair):
        position, chunk = pair
        results = []
        with closing(storage.Client(project=config.GCP_PROJECT)) as client:
            bucket = client.bucket(self.source)
            with tqdm(chunk, total=len(chunk), position=position) as progress:
                for f_name in progress:
                    with bucket.blob(f_name).open("r") as fp:
                        article = json.load(fp)
                        if self.filter_f(article):
                            results.append(article)
        return results


def get_work(source=config.ARTICLE_TARGET_BUCKET, filter_f=lambda article: True):
    with closing(storage.Client(project=config.GCP_PROJECT)) as client:
        tbd = [f.name for f in client.list_blobs(bucket_or_name=source)]
        workers = cpu_count() - 1
        tbd_chunks = np.array_split(tbd, workers)

    print("Processing {chuck_size} chunks".format(chuck_size=len(tbd_chunks)))
    final = []
    with mp.Pool(processes=workers, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as pool:
        for result in pool.imap_unordered(FilterArticle(filter_f=filter_f, source=source), enumerate(tbd_chunks)):
            final.extend(result)

    return final


def year_month_filter(article, year, month):
    timestamp = datetime.fromisoformat(article["published"])
    return timestamp.year == year and timestamp.month == month


if __name__ == "__main__":
    year = 2023
    month = 4

    def filter_article(article):
        return year_month_filter(article, year, month)

    sample = get_work(filter_f=filter_article)
    for s in sample:
        if s["source"] == "reuters" and len(s["title"]) < len(s["category"]):
            real_title = s["category"]
            s["category"] = s["title"]
            s["title"] = real_title

    final_df = pd.DataFrame(sample)

    target_url = "gs://{bucket}/{file}".format(bucket=config.ARTICLE_CONVERT_SUBSAMPLE_TARGET,
                                               file=config.ARTICLE_CONVERT_SUBSAMPLE_FILE)
    final_df.to_parquet(target_url.format(year=year, month=month), index=False)
