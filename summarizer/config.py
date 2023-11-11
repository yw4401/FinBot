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
ARTICLE_K = 3
QA_RESP_PROMPT = (
    "You are an AI assistant that will answer the given question in a concise way from the given context. "
    "You MUST only use the information given in the context, and not your prior knowledge. "
    "The response should be a well written, well formatted single paragraph. "
    "Do not include any newline or line breaks in the response."
    "If you cannot answer the question via the given context, respond with "
    '"Sorry, I cannot answer this question using the articles."\n\nBEGIN CONTEXT:\n{context}\n\n'
    'BEGIN QUESTION:\n{question}')
FUSION_PROMPT = ("You are an AI assistant that will help with optimizing queries for a retrieval augmented generation "
                 "system. You will be given a current query, the previous query before the current query, and the "
                 "previous response associated with the previous query. Then, you will first rewrite the current "
                 "query based on the previous query and response so that the meaning and intent of the rewritten query "
                 "does not depend on the previous query and response. Finally, you will also come up with 5 "
                 "optimized alternative queries with the same intent of the current query but asked in different ways. "
                 "\n\n{format_instructions}\n\nPrevious Query: {previous_query}\n\nPrevious Response: {response}\n\n"
                 "Current Query: {current_query}\n\nFollow the instructions in both content and formatting.")
FUSION_CHUNKS = 7


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
REWRITE_PROMPT = ("You are a helpful AI assistant that will rewrite a current query that depends on "
                  "a previous query and response so that the current query can be understood "
                  "without referring to the previous query and response. If the meaning of the current query does not "
                  "depend on the previous query and response, then repeat the content of the current query.\n\n"
                  "Examples:\n"
                  "Previous Query: What is the new LLM from OpenAI?\n\n"
                  "Response: GPT-4\n\n"
                  "Current Query: What does it do?\n"
                  "Re-written Version: What does GPT-4, the new LLM from OpenAI do?\n\nFollow the example, and begin.\n\n"
                  "Previous Query: {prev_query}\n\nResponse:\n{response}\n\n"
                  "Current Query: {current_query}")

# Model Deployment
SUM_API_SERVER = "http://summarizer/v1"
SUM_API_MODEL = "Open-Orca/Mistral-7B-OpenOrca"
QA_MODEL = "openai"
SUM_MODEL = "openai"
NER_MODEL = "openai"
KPI_MODEL = "openai"
REWRITE_MODEL = "openai"
