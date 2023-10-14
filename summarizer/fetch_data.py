import os
from pathlib import Path

from google.cloud import storage
import config
import shutil


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)


def create_directories():
    Path(config.TOPIC_SUMMARY_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.TOPIC_ARTICLES_INDICES_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.TOPIC_SUMMARY_INDEX_DIR).mkdir(parents=True, exist_ok=True)


def fetch_topic_summary(year, month):
    bucket_fname = config.TOPIC_SUMMARY_PATTERN.format(year=year, month=month)
    local_fname = Path(config.TOPIC_SUMMARY_DIR, config.TOPIC_SUMMARY_PATTERN.format(year=year, month=month))
    download_blob(config.TOPIC_SUMMARY_BUCKET, bucket_fname, local_fname)


def fetch_articles_indices(year, month):
    bucket_fname = config.TOPIC_ARTICLES_INDICES_FILE.format(year=year, month=month)
    local_fname = Path(config.TOPIC_ARTICLES_INDICES_DIR, bucket_fname)
    local_dirname = Path(config.TOPIC_ARTICLES_INDICES_DIR,
                         config.TOPIC_ARTICLES_INDICES_PATTERN.format(year=year, month=month))
    download_blob(config.TOPIC_ARTICLES_INDICES_BUCKET, bucket_fname, local_fname)
    shutil.unpack_archive(local_fname, local_dirname)
    os.remove(local_fname)


def fetch_topic_sum_index(year, month):
    bucket_fname = config.TOPIC_SUMMARY_INDEX_FILE.format(year=year, month=month)
    local_fname = Path(config.TOPIC_ARTICLES_INDICES_DIR, bucket_fname)
    local_dirname = Path(config.TOPIC_ARTICLES_INDICES_DIR,
                         config.TOPIC_SUMMARY_INDEX_PATTERN.format(year=year, month=month))
    download_blob(config.TOPIC_SUMMARY_INDEX_BUCKET, bucket_fname, local_fname)
    shutil.unpack_archive(local_fname, local_dirname)
    os.remove(local_fname)


def fetch_year_month(year, month):
    create_directories()
    fetch_topic_summary(year, month)
    fetch_articles_indices(year, month)
    fetch_topic_sum_index(year, month)


if __name__ == "__main__":
    year = 2023
    month = 4
    fetch_year_month(year, month)
