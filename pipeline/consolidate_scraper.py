import json
import logging
import re
from contextlib import closing
from collections import namedtuple
from datetime import datetime
from google.cloud import storage
import google.cloud.logging
from enum import Enum


class SummaryType(Enum):

    NULL = 0,
    BULLETS = 1,
    PLAIN = 2


BUCKET_IDX_FILE = "index.json"
Article = namedtuple('Article', ['category', 'title', 'published', 'body', 'summary', 'summary_type'])


def load_conversion_index(client, bucket="meta-info", conversion_index="scraper-markdown-index.json"):
    files = set([f.name for f in client.list_blobs(bucket_or_name=bucket)])
    if conversion_index not in files:
        return {
            "cnbc": 0,
            "reuters": 0,
            "standardized": 0
        }
    bucket = client.bucket(bucket)
    with bucket.blob(conversion_index).open("r") as fp:
        index = json.load(fp)
        if "cnbc" not in index:
            index["cnbc"] = 0
        if "reuters" not in index:
            index["reuters"] = 0
        if "standardized" not in index:
            index["standardized"] = 0
    return index


def get_upper_id(client, bucket):
    bucket = client.bucket(bucket)
    upper_id = -1
    while upper_id == -1:
        with bucket.blob(BUCKET_IDX_FILE).open("r") as fp:
            index = json.load(fp)
            if "counter" in index:
                upper_id = index["counter"]
    return upper_id


def write_conversion_index(client, index, bucket="meta-info", conversion_index="scraper-markdown-index.json"):
    bucket = client.bucket(bucket)
    with bucket.blob(conversion_index).open("w") as fp:
        json.dump(fp=fp, obj=index)


def normalize_text(text):
    return re.sub(r"\s+", " ", text).strip()


def extract_summary(summary_list):
    summary = ""
    summary_type = SummaryType.NULL

    if len(summary_list) == 1:
        summary = normalize_text(summary_list[0])
        if len(summary) > 0:
            summary_type = SummaryType.PLAIN
    else:
        for s in summary_list:
            summary = summary + "* " + normalize_text(s) + "\n"
        summary = summary.strip()
        summary_type = SummaryType.BULLETS
    return summary, summary_type


def convert_cnbc(cnbc_dict):
    title = cnbc_dict["title"]
    category = cnbc_dict["subsection"]
    published = datetime.strptime(cnbc_dict["published"], "%Y-%m-%dT%H:%M:%S%z")

    body = ""
    for d in cnbc_dict["body"]:
        if d["type"] == "Title":
            body = body + "## " + normalize_text(d["text"]) + "\n"
        if d["type"] == "NarrativeText":
            body = body + normalize_text(d["text"]) + "\n"
    body = body.strip()

    summary = ""
    summary_type = SummaryType.NULL
    if "summary" in cnbc_dict:
        sum_list = cnbc_dict["summary"]
        summary, summary_type = extract_summary(sum_list)
    return Article(
        category=category,
        title=title,
        published=published,
        body=body,
        summary=summary,
        summary_type=summary_type
    )


def convert_reuters(reuter_dict):
    title = reuter_dict["title"]
    category = reuter_dict["subsection"]
    published = datetime.strptime(reuter_dict["published"].strip(), "%B %d, %Y %I:%M %p")

    body = ""
    for b in reuter_dict["body"].split("\n\n"):
        body = body + normalize_text(b) + "\n"
    body = body.strip()

    summary = ""
    summary_type = SummaryType.NULL
    if "summary" in reuter_dict:
        sum_list = reuter_dict["summary"].split("\n\n")
        summary, summary_type = extract_summary(sum_list)

    return Article(
        category=category,
        title=title,
        published=published,
        body=body,
        summary=summary,
        summary_type=summary_type
    )


CONVERTER_REGISTRY = {
    "cnbc": ("cnbc-articles", convert_cnbc),
    "reuters": ("reuters-articles", convert_reuters)
}


def convert_raw_data(client, target, target_bucket_name, counter, start_id):
    source_bucket = CONVERTER_REGISTRY[target][0]
    upper_exc = get_upper_id(client, source_bucket)
    files = set([f.name for f in client.list_blobs(bucket_or_name=source_bucket)])
    bucket = client.bucket(source_bucket)
    target_bucket = client.bucket(target_bucket_name)

    for i in range(start_id, upper_exc):
        file_name = "%s.json" % i
        if file_name not in files:
            continue
        with bucket.blob(file_name).open("r") as fp:
            try:
                source_obj = json.load(fp)
            except Exception as e:
                logging.warning("Failed to read scraped article: %s. Reason: %s" % (file_name, str(e)))
                continue
            article = CONVERTER_REGISTRY[target][1](source_obj)
            output = {
                "source": target,
                "id": i,
                "category": article.category,
                "title": article.title,
                "published": article.published.isoformat(),
                "body": article.body,
                "summary": article.summary,
                "summary_type": article.summary_type.name
            }
        target_fname = "%s.json" % counter
        with target_bucket.blob(target_fname).open("w") as tfp:
            json.dump(output, tfp)
            counter = counter + 1

    return counter, upper_exc


TARGET_BUCKET = "markdown-converged"


if __name__ == "__main__":
    log_client = google.cloud.logging.Client(project="msca310019-capstone-f945")
    log_client.setup_logging()
    logging.getLogger().setLevel(logging.INFO)
    with closing(storage.Client(project="msca310019-capstone-f945")) as client:
        conversion_idx = load_conversion_index(client)
        counter = conversion_idx["standardized"]
        logging.info("Starting conversion from idx: %s" % counter)

        for target in CONVERTER_REGISTRY:
            logging.info("Starting to process %s from idx: %s" % (target, conversion_idx[target]))
            try:
                counter, next_start = convert_raw_data(client, target, TARGET_BUCKET, counter, conversion_idx[target])
            except Exception as e:
                logging.error("Failed to process: %s. Reason: %s" % (target, e))
                continue
            conversion_idx[target] = next_start
            conversion_idx["standardized"] = counter
            write_conversion_index(client, conversion_idx)
            logging.info("Ended processing %s up to idx: %s" % (target, conversion_idx[target]))
            logging.info("Starting next batch from idx: %s" % counter)
