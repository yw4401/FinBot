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
TOPIC_K = 2

# Article QA Response
ARTICLE_K = 7
QA_RESP_PROMPT = (
    "You are an AI assistant that will answer the given question in a concise way from the given context. "
    "You MUST only use the information given in the context, and not your prior knowledge. "
    "The response should be a well written, well formatted single paragraph. "
    "DO NOT assume any facts unless the context explicitly provides it. "
    "DO NOT assume that the information in the given question is correct. "
    "If you cannot answer the question via the given context, respond with "
    '"Impossible to answer with given information"\n\nBEGIN CONTEXT:\n{context}\n\n'
    'BEGIN QUESTION:\n{question}')
FUSION_SYSTEM_PROMPT = (
    "You are a helpful AI assistant that will help with optimizing queries for a retrieval augmented generation "
    "system. Given a query, you will come up with 5 alternative optimized queries that are written differently but "
    "preserves the same intent. Follow the following format: {format_instructions}")
FUSION_USER_PROMPT = "Query: {query}"
REWRITE_SYSTEM_PROMPT = "You are a helpful AI assistant."
REWRITE_USER_PROMPT = ("Rephrase the following query as a question such that it can be understood without reading "
                       "the conversation so far. The rephrased version should capture the intent of the query while "
                       "being self-contained with respect to context:\n{query}")
FUSION_CHUNKS = 7
FUSION_SRC_CHUNKS = 50

# Article Summary Response
TOPIC_SUM_K = 5
TOPIC_SUM_TOP_K = 2
TOPIC_SUM_CHUNKS = 7
TOPIC_SUM_MISTRAL_PROMPT = """<|im_start|>system
Summarize the key-points from the given context. The information in the summary should include, but should not be limited to information that can help answer the given question. Be concise if possible. Respond with "Impossible to answer with given information" if the context does not contain information that can answer the given question.
<|im_end|>
<|im_start|>user
BEGIN CONTEXT:\n{context}\n\nBEGIN QUESTION:\n{question}\n<|im_end|>
<|im_start|>assistant: """
QA_LLAMA_PROMPT = """[INST] <<SYS>>
You are an AI assistant that will answer the given question in a concise way from the given context. You MUST only use the information given in the context, and not your prior knowledge. The response should be a well written, well formatted single paragraph. DO NOT assume any facts unless the context explicitly provides it. DO NOT assume that the information in the given question is correct. If you cannot answer the question via the given context, respond with "Impossible to answer with given information"
<</SYS>>

BEGIN CONTEXT:\n{context}\n\nBEGIN QUESTION:\n{question}\n [/INST] """
TOPIC_SUM_GENERIC_PROMPT = ('Summarize the key-points from the given context. '
                            'The information in the summary should include, '
                            'but should not be limited to information that can help answer the given question. '
                            'Be concise if possible. '
                            'Respond with "IMPOSSIBLE" if the context does not contain information that can '
                            'answer the given question. The format of the response should be the same as the following example:\n'
                            "EXAMPLE:\nAnthropic’s AI chatbot Claude is posting lyrics to popular songs, lawsuit claims\n"
                            "* Universal Music has sued Anthropic, the AI startup, over “systematic and widespread infringement of their copyrighted song lyrics,” per a filing Wednesday in a Tennessee federal court.\n"
                            "* Anthropic, the $4.1 billion startup behind Claude the chatbot, was founded in 2021 by former OpenAI research executives.\n"
                            "* Other music publishers, such as Concord and ABKCO, were also named as plaintiffs\n\n"
                            'Text: {context}\n\nQuery: {question}\n')

# KPI Extraction
NER_RESPONSE_PROMPT = ("You are a helpful AI assistant that will provide the stock tickers associated with companies "
                       "mentioned in the given text. Then, you will order the identified ticker symbols from the "
                       "most relevant to the least relevant with respect to the given query. The ticker symbol should "
                       "be the ones that can be used to look up stock prices on Yahoo Finance. "
                       "Thus,  if the text explicitly mentions a stock ticker symbol, "
                       "then you should convert the symbol to one "
                       "that can be used by Yahoo Finance."
                       "\n{format_instructions}\n\n"
                       "Text:\n{text}\n\nQuery:{query}")
KPI_PROMPT = ("You are a helpful AI assistant that will identify the top relevant KPI/metrics for a given stock. "
              "The relevance of a KPI depends on whether the user would be interested in knowing it given the query "
              "and the description of the company. "
              "You should group the relevant KPIs into logical sections with an "
              "appropriate title with appropriate capitalization.\n"
              "KPIs: {kpi}\n\nCompany: {response}\n\nQuery: {query}\n\n"
              "{format_instructions}")

# Model Deployment
SUM_API_SERVER = "http://summarizer/v1"
SUM_API_MODEL = "Open-Orca/Mistral-7B-OpenOrca"
QA_API_SERVER = "http://qa-vm/v1"
QA_API_MODEL = "meta-llama/Llama-2-13b-chat-hf"
QA_MODEL = "custom"
SUM_MODEL = "custom"
NER_MODEL = "vertexai"
KPI_MODEL = "vertexai"
REWRITE_MODEL = "openai"
