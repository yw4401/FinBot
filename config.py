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

# Co-reference Resolution
ARTICLE_COREF_IDX = "coref-index.json"
ARTICLE_COREF_TARGET_BUCKET = "markdown-corref"
ARTICLE_COREF_SRC_BUCKET = ARTICLE_TARGET_BUCKET
ARTICLE_COREF_MOD_URL = "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz"
ARTICLE_COREF_SPACY_MOD = "en_core_web_sm"
