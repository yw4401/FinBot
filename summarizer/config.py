GCP_PROJECT = "msca310019-capstone-f945"

# Data Locations
TOPIC_SUMMARY_DIR = "./topics/summaries/"
TOPIC_ARTICLES_INDICES_DIR = "./topics/topic_indices/"
TOPIC_SUMMARY_INDEX_DIR = "./topics/topic_indices/"
TOPIC_SUMMARY_PATTERN = "topicsum-{year}-{month}.parquet"
TOPIC_ARTICLES_INDICES_PATTERN = "articles-{year}-{month}/"
TOPIC_SUMMARY_INDEX_PATTERN = "topic-{year}-{month}/"

# Fetch Locations
TOPIC_SUMMARY_BUCKET = "scraped-news-article-data-null"
TOPIC_SUMMARY_FILE = "topicsum-{year}-{month}.parquet"
TOPIC_ARTICLES_INDICES_BUCKET = TOPIC_SUMMARY_BUCKET
TOPIC_ARTICLES_INDICES_FILE = "article-chroma-{year}-{month}.zip"
TOPIC_SUMMARY_INDEX_BUCKET = TOPIC_SUMMARY_BUCKET
TOPIC_SUMMARY_INDEX_FILE = "topic-chroma-{year}-{month}.zip"

# Embeddings
OPENAI_API = "./key"
FILTER_EMBEDDINGS = "shilongdai/finember"
TOPIC_COLLECTION = "topics"
ARTICLE_COLLECTION = "articles"

# Elasticsearch
ES_CLOUD_ID_PATH = "./es_id"
ES_KEY_PATH = "./es_key"
ES_TOPIC_INDEX = "topics"
ES_ARTICLE_INDEX = "articles"
ES_TOPIC_VECTOR_FIELD = "description_embedding"
ES_TOPIC_FIELD = "description"
ES_ARTICLE_VECTOR_FIELD = "chunk_text_embedding"
ES_ARTICLE_FIELD = "chunk_text"

# Topic Filtering
TOPIC_FILTER_RAW_TEMPLATE = "Which topics are relevant to the query: {question}"
TOPIC_FILTER_FORMAT_SYSTEM = "You are an AI assistant that will format given text. " \
                             "The formatted output should only contain the topic numbers as integers.\n" \
                             "{format_instructions}"
TOPIC_FILTER_FORMAT_USER = "{result}"
TOPIC_K = 3

# Article QA Response
ARTICLE_K = 2

# Article Summary Response
TOPIC_SUM_K = 5
TOPIC_SUM_TOP_K = 2
TOPIC_SUM_CHUNKS = 14
TOPIC_SUM_PROMPT = """<|im_start|>system
Summarize the key-points from the given context. The information in the summary should include, but should not be limited to information that can help answer the given question. Be concise if possible. Respond with "IMPOSSIBLE" if the context does not contain information that can answer the given question.
<|im_end|>
<|im_start|>user
BEGIN CONTEXT:\n{context}\n\nBEGIN QUESTION:\n{question}\n<|im_end|>
<|im_start|>assistant: """

# Model Deployment
SUM_API_SERVER = "http://10.142.190.206/v1"
SUM_API_MODEL = "Open-Orca/Mistral-7B-OpenOrca"

