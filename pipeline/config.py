GCP_PROJECT = "msca310019-capstone-f945"

# Article Ingestion
ARTICLE_TARGET_BUCKET = "markdown-converged"
ARTICLE_CONVERT_META_BUCKET = "meta-info"
ARTICLE_CONVERT_META_IDX = "scraper-markdown-index.json"
ARTICLE_BUCKET_IDX = "index.json"
ARTICLE_CONVERT_CHUNK = 100
ARTICLE_CONVERT_CNBC_DATE = "%Y-%m-%dT%H:%M:%S%z"
ARTICLE_CONVERT_REUTER_DATE = "%B %d, %Y %I:%M %p"
ARTICLE_CONVERT_NYT_DATE = "%Y-%m-%dT%H:%M:%S%z"
ARTICLE_CONVERT_SUBSAMPLE_TARGET = "scraped-news-article-data-null"
ARTICLE_CONVERT_SUBSAMPLE_FILE = "subsample-{year}-{month}.parquet"
ARTICLE_CONVERT_SUBSAMPLE_IDX = "scraper-subsample.json"

# Fine-tuning Ingestion
FINE_TUNE_TARGET_BUCKET = "scraped-news-article-data-null"
FINE_TUNE_FILE_PATTERN = "fine-tune-summary-{split}.parquet"
FINE_TUNE_FILE_CHUNK = 64

# Co-reference Resolution
ARTICLE_COREF_SRC_BUCKET = ARTICLE_CONVERT_SUBSAMPLE_TARGET
ARTICLE_COREF_MOD_URL = "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz"
ARTICLE_COREF_SPACY_MOD = "en_core_web_sm"
ARTICLE_COREF_TARGET_BUCKET = ARTICLE_CONVERT_SUBSAMPLE_TARGET
ARTICLE_COREF_FILE_PATTERN = "corref-{year}-{month}.parquet"

# Topic Extraction
TOPIC_EMBEDDING = "all-MiniLM-L6-v2"
TOPIC_UMAP_NEIGHBORS = 15
TOPIC_UMAP_COMPONENTS = 5
TOPIC_UMAP_MIN_DIST = 0
TOPIC_UMAP_METRIC = "euclidean"
TOPIC_HDBSCAN_MIN_SIZE = 5
TOPIC_HDBSCAN_METRIC = "euclidean"
TOPIC_SUBSAMPLE_TARGET = ARTICLE_CONVERT_SUBSAMPLE_TARGET
TOPIC_SUBSAMPLE_FILE = "topic-{year}-{month}.parquet"

# Topic Summarization
TOPIC_SUM_KIND = "openai"
TOPIC_SUM_MODEL_PARAMS = {
    "api_key": "../key"
}
TOPIC_SUM_TARGET = "scraped-news-article-data-null"
TOPIC_SUM_TARGET_FILE = "topicsum-{year}-{month}.parquet"

# NER
NER_SPACY_MOD = "en_core_web_md"
NER_TARGET_BUCKET = ARTICLE_COREF_TARGET_BUCKET
NER_TARGET_PATTERN = "ner-{year}-{month}.parquet"

# FAISS Articles
ARTICLE_FAISS_TEMP_DIRECTORY = "./article_db"
ARTICLE_FAISS_EMBEDDING = "thenlper/gte-base"
ARTICLE_DB_COLLECTION = "articles"
TOPICS_DB_COLLECTION = "topics"
ARTICLE_FAISS_TARGET = "scraped-news-article-data-null"
ARTICLE_FAISS_FILE = "article-chroma-{year}-{month}.zip"
ARTICLE_FAISS_BATCH = 16
ARTICLE_FAISS_PROCESSES = None

# FAISS Topics
TOPIC_FAISS_EMBEDDING = ARTICLE_FAISS_EMBEDDING
TOPIC_FAISS_TARGET = "scraped-news-article-data-null"
TOPIC_FAISS_FILE = "topic-chroma-{year}-{month}.zip"
TOPIC_FAISS_BATCH = 16
