import os
import shutil
from multiprocessing import Pool
from multiprocessing import cpu_count
from pathlib import Path

import pandas as pd
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
from llama_index import ServiceContext, LangchainEmbedding
from llama_index.storage.storage_context import StorageContext
from tqdm import tqdm

import config
from common import upload_blob

os.environ["OPENAI_API_KEY"] = "None"


def build_topic_idx(data_dir, persist_dir, t):
    topic_data_dir = Path(data_dir, str(t))
    topic_index_dir = Path(persist_dir, str(t))
    topic_index_dir.mkdir(parents=True, exist_ok=True)

    embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name=config.ARTICLE_FAISS_EMBEDDING))
    service_context = ServiceContext.from_defaults(chunk_size_limit=512, embed_model=embed_model)
    storage_context = StorageContext.from_defaults()
    documents = SimpleDirectoryReader(topic_data_dir).load_data()

    _ = GPTVectorStoreIndex.from_documents(documents, service_context=service_context,
                                           storage_context=storage_context)
    storage_context.persist(persist_dir=topic_index_dir)
    return topic_index_dir


def prepare_articles(article_df, temp_path):
    topics = list(set(article_df.topic.unique()) - {-1})
    for t in topics:
        topic_path = Path(temp_path, "articles", str(t))
        topic_path.mkdir(parents=True, exist_ok=True)
        topic_article_df = article_df.loc[article_df.topic == t]
        topic_article_df = topic_article_df.sort_values(by="probability", ascending=False)

        for i in range(len(topic_article_df.index)):
            doc_path = Path(temp_path, "articles", str(t), str(i) + ".txt")
            with open(doc_path, "w") as fp:
                fp.write(topic_article_df["body"].iloc[i])
    return topics


class StarFunc:

    def __init__(self, func):
        self.func = func

    def __call__(self, args):
        result = self.func(*args)
        return result


if __name__ == "__main__":
    year = 2023
    month = 4

    src_url = "gs://{bucket}/{file}".format(bucket=config.TOPIC_SUBSAMPLE_TARGET,
                                            file=config.TOPIC_SUBSAMPLE_FILE)
    article_df = pd.read_parquet(src_url.format(year=year, month=month))
    topics = prepare_articles(article_df, config.ARTICLE_FAISS_TEMP_DIRECTORY)

    with tqdm(total=len(topics)) as progress:
        with Pool(processes=cpu_count() - 1) as pool:
            for _ in pool.imap_unordered(StarFunc(build_topic_idx),
                                         [(Path(config.ARTICLE_FAISS_TEMP_DIRECTORY, "articles"),
                                           Path(config.ARTICLE_FAISS_TEMP_DIRECTORY, "index"),
                                           t) for t in topics],
                                         chunksize=config.ARTICLE_FAISS_BATCH):
                progress.update(1)

    final_dir = Path(config.ARTICLE_FAISS_TEMP_DIRECTORY, "index")
    final_file = Path(config.ARTICLE_FAISS_TEMP_DIRECTORY, "target")
    dest_file = config.ARTICLE_FAISS_FILE.format(year=year, month=month)
    shutil.make_archive(final_file, 'zip', final_dir)
    upload_blob(bucket_name=config.ARTICLE_FAISS_TARGET,
                source_file_name=Path(config.ARTICLE_FAISS_TEMP_DIRECTORY, "target.zip"),
                destination_blob_name=dest_file)
    shutil.rmtree(config.ARTICLE_FAISS_TEMP_DIRECTORY)
