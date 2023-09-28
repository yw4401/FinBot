import json
from contextlib import closing
import pandas as pd
from tqdm import tqdm
import numpy as np

import google.cloud.storage as storage

import config


def fetch_articles(bucket):
    with closing(storage.Client(project=config.GCP_PROJECT)) as count_client:
        tbd = [f.name for f in count_client.list_blobs(bucket_or_name=bucket)]

    def generator():
        with closing(storage.Client(project=config.GCP_PROJECT)) as client:
            bkt = client.bucket(bucket)
            for fname in tbd:
                with bkt.blob(fname).open("r") as fp:
                    try:
                        article_dict = json.load(fp)
                        yield fname, article_dict
                    except:
                        continue

    return len(tbd), generator


if __name__ == "__main__":
    articles = []
    sample = None
    cur_max = -1
    counter = 0
    amounts, generator = fetch_articles(config.FINE_TUNE_SRC_BUCKET)
    target_url = "gs://{bucket}/{pattern}".format(bucket=config.FINE_TUNE_TARGET_BUCKET,
                                                  pattern=config.FINE_TUNE_FILE_PATTERN)
    target_url = target_url.format(id=cur_max)
    with tqdm(total=amounts) as progress:
        for _, d in generator():
            try:
                if sample and counter > sample:
                    continue
                if d["summary"].strip() == "":
                    continue
                articles.append(d)
                cur_max = max(int(d["id"]), cur_max)
                counter += 1
            except:
                pass
            finally:
                progress.update(1)

    df = pd.DataFrame(articles)
    print(df.head())
    print("Uploading article snapshots")
    df.to_parquet(target_url, index=False)
