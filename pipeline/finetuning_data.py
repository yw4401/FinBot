import json
from contextlib import closing
import pandas as pd
from tqdm import tqdm

import google.cloud.storage as storage

import config


def fetch_articles(bucket):
    with closing(storage.Client(project=config.GCP_PROJECT)) as count_client:
        tbd = set([f.name for f in count_client.list_blobs(bucket_or_name=bucket)])

    def generator():
        with closing(storage.Client(project=config.GCP_PROJECT)) as client:
            bkt = client.bucket(bucket)
            for fname in tbd:
                with bkt.blob(fname).open("r") as fp:
                    try:
                        article_dict = json.load(fp)
                        yield int(fname.split(".")[0]), article_dict
                    except:
                        continue

    return len(tbd), generator


if __name__ == "__main__":
    articles = []
    cur_max = -1
    counter = 0
    amounts, generator = fetch_articles(config.FINE_TUNE_SRC_BUCKET)
    with tqdm(total=amounts) as progress:
        for id_num, d in generator():
            if d["summary"].strip() == "":
                continue
            articles.append(d)
            cur_max = max(id_num, cur_max)
            progress.update(1)

    df = pd.DataFrame(articles)
    print(df.head())
    print("Uploading article snapshots")
    target_url = "gs://{bucket}/{pattern}".format(bucket=config.FINE_TUNE_SRC_BUCKET,
                                                  pattern=config.FINE_TUNE_FILE_PATTERN)
    target_url = target_url.format(id=cur_max)
    df.to_parquet(target_url, index=False)

