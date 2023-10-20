from functools import reduce

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
from consolidate_scraper import write_conversion_index


class DownloadArticle:

    def __init__(self, source):
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
                        results.append(article)
        return results


def load_subsample_index(client, bucket=config.ARTICLE_CONVERT_META_BUCKET,
                         conversion_index=config.ARTICLE_CONVERT_SUBSAMPLE_IDX):
    files = set([f.name for f in client.list_blobs(bucket_or_name=bucket)])
    if conversion_index not in files:
        return {
            "subsampled": 0,
            "files": []
        }
    bucket = client.bucket(bucket)
    with bucket.blob(conversion_index).open("r") as fp:
        index = json.load(fp)
    return index


def get_work(source=config.ARTICLE_TARGET_BUCKET):
    with closing(storage.Client(project=config.GCP_PROJECT)) as client:
        idx = load_subsample_index(client)
        cur_id = idx["subsampled"]
        tbd = [f.name for f in client.list_blobs(bucket_or_name=source) if int(f.name.split(".")[0]) >= cur_id]
        if len(tbd) == 0:
            return [], idx
        next_id = max([int(s.split(".")[0]) for s in tbd]) + 1
        idx["subsampled"] = next_id
        workers = cpu_count() - 1
        tbd_chunks = np.array_split(tbd, workers)

    print("Total: {amount} articles".format(amount=len(tbd)))
    print("Processing {chuck_size} chunks".format(chuck_size=len(tbd_chunks)))
    final = []
    with mp.Pool(processes=workers, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as pool:
        for result in pool.imap_unordered(DownloadArticle(source=source), enumerate(tbd_chunks)):
            final.extend(result)

    return final, idx


def reduce_work(target_dict, next_dict):
    timestamp = datetime.fromisoformat(next_dict["published"])
    key_tuple = (timestamp.year, timestamp.month)

    if next_dict["source"] == "reuters" and len(next_dict["title"]) < len(next_dict["category"]):
        real_title = next_dict["category"]
        next_dict["category"] = next_dict["title"]
        next_dict["title"] = real_title

    if key_tuple not in target_dict:
        target_dict[key_tuple] = [next_dict]
    else:
        target_dict[key_tuple].append(next_dict)
    return target_dict


if __name__ == "__main__":
    sample, new_idx = get_work()
    print()
    categorized_articles = reduce(reduce_work, [{}] + sample)

    for year, month in categorized_articles:
        new_df = pd.DataFrame(categorized_articles[(year, month)])
        target_url = "gs://{bucket}/{file}".format(bucket=config.ARTICLE_CONVERT_SUBSAMPLE_TARGET,
                                                   file=config.ARTICLE_CONVERT_SUBSAMPLE_FILE)
        target_url = target_url.format(year=year, month=month)
        print(f"Writing to {target_url}")
        try:
            old_df = pd.read_parquet(target_url)
            final_df = pd.concat([old_df, new_df], ignore_index=True).reset_index(drop=True)
        except FileNotFoundError:
            final_df = new_df
        final_df = final_df.drop_duplicates(subset=["url"])
        final_df.to_parquet(target_url, index=False)
        new_idx["files"].append(target_url)
    with closing(storage.Client(project=config.GCP_PROJECT)) as client:
        write_conversion_index(client=client, index=new_idx, conversion_index=config.ARTICLE_CONVERT_SUBSAMPLE_IDX)