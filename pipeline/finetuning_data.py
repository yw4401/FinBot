import re
from contextlib import closing
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import google.cloud.storage as storage

import config
from select_data import load_subsample_index


prefix_bullets = "summarize in bullet points: "
prefix_plain = "summarize as paragraph: "


def prepend_prefix(row):
    if row["summary_type"] == "BULLETS":
        return prefix_bullets + row["body"]
    elif row["summary_type"] == "PLAIN":
        return prefix_plain + row["body"]
    else:
        raise ValueError("Type")


def get_data_sets_df(sample_df, test_instances=1000):
    sample_df = sample_df.sample(frac=1, random_state=93).reset_index(drop=True)
    clean_regex = re.compile(r"\*[\s\n]*(?=\*)")
    sample_df["summary"] = sample_df.summary.apply(lambda s: clean_regex.sub(" ", s).strip())
    sample_df["summary"] = sample_df.title.str.strip() + "\n" + sample_df.summary
    sample_df["summary"] = sample_df.summary.str.strip()
    sample_df["body"] = sample_df.apply(prepend_prefix, axis=1)
    sample_df = sample_df[["body", "summary"]]

    train_df, test_df = train_test_split(sample_df, test_size=test_instances)
    return train_df, test_df


if __name__ == "__main__":
    articles = []
    with closing(storage.Client(project=config.GCP_PROJECT)) as client:
        idx = load_subsample_index(client=client)
    target_url = "gs://{bucket}/{pattern}".format(bucket=config.FINE_TUNE_TARGET_BUCKET,
                                                  pattern=config.FINE_TUNE_FILE_PATTERN)
    with tqdm(total=len(idx["files"])) as progress:
        for f in idx["files"]:
            temp_df = pd.read_parquet(f)
            articles.append(temp_df)
            progress.update(1)

    df = pd.concat(articles, ignore_index=True)
    df = df.loc[df.summary_type != "NULL"]
    df = df.drop_duplicates(subset=["body"])
    print(df.head())
    print("Uploading article snapshots")
    train, test = get_data_sets_df(df)
    train.to_parquet(target_url.format(split="train"), index=False)
    test.to_parquet(target_url.format(split="test"), index=False)
