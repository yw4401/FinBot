GCP_PROJECT = "msca310019-capstone-f945"

# Article Ingestion
ARTICLE_TARGET_BUCKET = "consolidated-articles"
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
TOPIC_SUM_TARGET = "scraped-news-article-data-null"
TOPIC_SUM_TARGET_FILE = "topicsum-{year}-{month}.parquet"
TOPIC_SUM_HF_PROMPT = "summarize in bullet points:\n{text}"
TOPIC_SUM_LC_SYSTEM_PROMPT = "A list of news article titles with the published time is given below. " + \
                             "Using only the provided information, summarize the theme of the titles such that it will be easy to answer investing related questions from the summary. " + \
                             'Before providing the summary, explain how the summary can be used to answer investing related questions. ' + \
                             "Be specific about the dates and entities involved. " + \
                             "Be concise in writing the summary, but try not to omit important details. " + \
                             'Do not use vague terms such as "past few months", "various companies", or "the disease". Use the actual names if possible. ' + \
                             'If a clear theme relevant to investing is not present, maintain the specified format, use "SUMMARY: NO THEME" as the summary. ' + \
                             'The format of the output should be:\nEXPLANATION:\nthe explanation\nSUMMARY:\nthe summary'
TOPIC_SUM_LC_USER_PROMPT = "{text}"
TOPIC_SUM_LC_REGEX = r"SUMMARY:(?P<summary>(.|\n)+)"
TOPIC_SUM_LC_DEFAULT = "NO THEME"

# NER
NER_SPACY_MOD = "en_core_web_md"

# Chroma Articles
ARTICLE_SPLITTER_TOKENIZER = "google/flan-t5-xxl"
ARTICLE_SPLITTER_CHUNK_SIZE = 256
ARTICLE_SPLITTER_CHUNK_OVERLAP = 0
ARTICLE_FAISS_TEMP_DIRECTORY = "./article_db"
ARTICLE_FAISS_EMBEDDING = "../embeddings"
ARTICLE_DB_COLLECTION = "articles"
TOPICS_DB_COLLECTION = "topics"
ARTICLE_FAISS_TARGET = "scraped-news-article-data-null"
ARTICLE_FAISS_FILE = "article-chroma-{year}-{month}.zip"
ARTICLE_FAISS_BATCH = 16
ARTICLE_FAISS_PROCESSES = None

# Chroma Topics
TOPIC_FAISS_EMBEDDING = ARTICLE_FAISS_EMBEDDING
TOPIC_FAISS_TARGET = "scraped-news-article-data-null"
TOPIC_FAISS_FILE = "topic-chroma-{year}-{month}.zip"
TOPIC_FAISS_BATCH = 16

# Elasticsearch Topics
ES_CLOUD_ID_PATH = "../es_id"
ES_KEY_PATH = "../es_key"
ES_TOPIC_INDEX = "topics"
ES_TOPIC_MAPPING = {
    "properties": {
        "description": {"type": "text", "index": True},
        "description_embedding": {
            "type": "dense_vector",
            "dims": 1024,
            "index": True,
            "similarity": "cosine"
        },
        "metadata": {
            "properties": {
                "topic": {"type": "integer", "index": True},
                "created_at": {"type": "date", "index": True}
            }
        }
    }
}

# Elasticsearch Articles
ES_ARTICLE_INDEX = "articles"
ES_ARTICLE_ENTITIES = {"ORG", "PERSON", "GPE", "PRODUCT", "LAW"}
ES_ARTICLES_MAPPING = {
    "properties": {
        "chunk_text": {"type": "text", "index": True},
        "chunk_text_embedding": {
            "type": "dense_vector",
            "dims": 1024,
            "index": True,
            "similarity": "cosine"
        },
        "metadata": {
            "properties": {
                "entities": {"type": "text", "index": True},
                "topic": {"type": "integer", "index": True},
                "published_at": {"type": "date", "index": True},
            }
        }
    }
}

# AI Assisted Summary Augmentation
VERTEX_AI_KEY_PATH = "../nlp-final-386206-e7043b88437d.json"
VERTEX_AI_PROJECT = "nlp-final-386206"
SUMMARY_AUG_SYSTEM = "You are a helpful AI assistant that will come up with a good question that can be answered by the given text, " \
                     "and a good question that cannot be answered by the text. " \
                     "The answer to the answerable question must be from the information in the provided text. " \
                     "The question must be something that a retail investor is likely going to ask. " \
                     "The output should be formatted as:\n" + \
                     """Response:
REASON (Unanswerable): reason why a retail investor would ask the question that cannot be answered.
UNANSWERABLE: reason why the question cannot be answered with only information from the text.
QUESTION (Unanswerable): the unanswerable question.

REASON (answerable): reason why a retail investor would ask the question that can be answered.
ANSWER: answer to the answerable question from the text.
QUESTION (answerable): the answerable question.

An example:
Text: 
Anthropic’s AI chatbot Claude is posting lyrics to popular songs, lawsuit claims
* Universal Music has sued Anthropic, the AI startup, over “systematic and widespread infringement of their copyrighted song lyrics,” per a filing Wednesday in a Tennessee federal court.
* Anthropic, the $4.1 billion startup behind Claude the chatbot, was founded in 2021 by former OpenAI research executives.
* Other music publishers, such as Concord and ABKCO, were also named as plaintiffs
Response:

REASON (Unanswerable): Retail investors would want to estimate the potential financial repercussions of the lawsuit on Anthropic in order to assess the company's financial stability and investment prospects.
UNANSWERABLE: The text only mention that a lawsuit occurred, but does not mention any specific consequences of the lawsuit
QUESTION: What potential financial damages or penalties is Anthropic facing as a result of the lawsuit filed by Universal Music and other plaintiffs?

REASON: Since the AI industry and Anthropic are both relatively new, retail investors may want to understand the regulatory and legal risks associated before making any move.
ANSWER: Anthropic is being sued by music publishers including Universal Music, Concord, and ABKCO over copyrighted lyrics generated by Claude.
QUESTION: What are some legal risks that Anthropic is currently facing?

Follow the example. Begin.
"""
SUMMARY_AUG_USER = "{input_text}"
SUMMARY_AUG_PARSE_SYSTEM = "You are a helpful AI assistant that will convert natural language into a machine parsable format. {format_instructions}"
SUMMARY_AUG_PARSE_EXAMPLE = {"raw": """REASON (Unanswerable): Retail investors would want to estimate the potential financial repercussions of the lawsuit on Anthropic in order to assess the company's financial stability and investment prospects.
UNANSWERABLE: The text only mention that a lawsuit occurred, but does not mention any specific consequences of the lawsuit
QUESTION: What potential financial damages or penalties is Anthropic facing as a result of the lawsuit filed by Universal Music and other plaintiffs?

REASON: Since the AI industry and Anthropic are both relatively new, retail investors may want to understand the regulatory and legal risks associated before making any move.
ANSWER: Anthropic is being sued by music publishers including Universal Music, Concord, and ABKCO over copyrighted lyrics generated by Claude.
QUESTION: What are some legal risks that Anthropic is currently facing?""",
                             "output": """{"answerable": "What are some legal risks that Anthropic is currently facing?", "unanswerable": "What potential financial damages or penalties is Anthropic facing as a result of the lawsuit filed by Universal Music and other plaintiffs?"}"""}
SUMMARY_AUG_PARSE_USER = "{text}"
SUMMARY_AUG_MODEL = "chat-bison"
SUMMARY_AUG_TEMPERATURE = 0
