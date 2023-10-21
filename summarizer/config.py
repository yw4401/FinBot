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
TOPIC_EMBEDDING = "thenlper/gte-base"
TOPIC_COLLECTION = "topics"
ARTICLE_COLLECTION = "articles"

# Topic Filtering
TOPIC_FILTER_RAW_TEMPLATE = "Which topics are relevant to the query: {question}"
TOPIC_FILTER_FORMAT_SYSTEM = "You are an AI assistant that will format given text. " \
                             "The formatted output should only contain the topic numbers as integers.\n" \
                             "{format_instructions}"
TOPIC_FILTER_FORMAT_USER = "{result}"
TOPIC_K = 3

# Article Response
ARTICLE_K = 2
