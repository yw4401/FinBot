GCP_PROJECT = "msca310019-capstone-f945"

# Article Ingestion
ARTICLE_INGEST_MAX_DAYS = 30 * 7
ARTICLE_CONVERT_CHUNK = 100
ARTICLE_CONVERT_CNBC_DATE = "%Y-%m-%dT%H:%M:%S%z"
ARTICLE_CONVERT_REUTER_DATE = "%B %d, %Y %I:%M %p"
ARTICLE_CONVERT_NYT_DATE = "%Y-%m-%dT%H:%M:%S%z"

# Fine-tuning Ingestion
FINE_TUNE_TARGET_BUCKET = "scraped-news-article-data-null"
FINE_TUNE_FILTERED = "fine-tune-filtered.parquet"
FINE_TUNE_COREF = "fine-tune-coref.parquet"
FINE_TUNE_FILE_PATTERN = "fine-tune-summary-{split}.parquet"

# Co-reference Resolution
ARTICLE_COREF_MOD_URL = "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz"
ARTICLE_COREF_SPACY_MOD = "en_core_web_sm"

# Topic Extraction
TOPIC_BUCKET = "topic-models-null"
TOPIC_TTL_DAY = 7
TOPIC_FIT_RANGE_DAY = ARTICLE_INGEST_MAX_DAYS
TOPIC_EMBEDDING = "all-MiniLM-L6-v2"
TOPIC_UMAP_NEIGHBORS = 15
TOPIC_UMAP_COMPONENTS = 5
TOPIC_UMAP_MIN_DIST = 0
TOPIC_UMAP_METRIC = "euclidean"
TOPIC_HDBSCAN_MIN_SIZE = 5
TOPIC_HDBSCAN_METRIC = "euclidean"

# Topic Summarization
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
ARTICLE_SPLITTER_TOKENIZER = "Open-Orca/Mistral-7B-OpenOrca"
ARTICLE_SPLITTER_CHUNK_SIZE = 512
ARTICLE_SPLITTER_CHUNK_OVERLAP = 64
ARTICLE_FAISS_TEMP_DIRECTORY = "./article_db"
ARTICLE_FAISS_EMBEDDING = "shilongdai/finember"
ARTICLE_DB_COLLECTION = "articles"
TOPICS_DB_COLLECTION = "topics"
ARTICLE_FAISS_TARGET = "scraped-news-article-data-null"
ARTICLE_FAISS_FILE = "article-chroma-{year}-{month}.zip"
ARTICLE_FAISS_BATCH = 16
ARTICLE_FAISS_PROCESSES = None

# Chroma Topics
TOPIC_FAISS_EMBEDDING = ARTICLE_FAISS_EMBEDDING
TOPIC_EMBED_TOP_THRESHOLD = 0.8
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
                "model": {"type": "integer", "index": True},
                "recency": {"type": "date", "index": True}
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
                "model": {"type": "integer", "index": True},
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

Follow the example. Begin.
"""
SUMMARY_AUG_USER = "{input_text}"
SUMMARY_AUG_EXAMPLES = [
    {
        "user": "Anthropic’s AI chatbot Claude is posting lyrics to popular songs, lawsuit claims\n"
                "* Universal Music has sued Anthropic, the AI startup, over “systematic and widespread infringement of their copyrighted song lyrics,” per a filing Wednesday in a Tennessee federal court."
                "* Anthropic, the $4.1 billion startup behind Claude the chatbot, was founded in 2021 by former OpenAI research executives.\n"
                "* Other music publishers, such as Concord and ABKCO, were also named as plaintiffs",
        "assistant": "REASON (Unanswerable): Retail investors would want to estimate the potential financial repercussions of the lawsuit on Anthropic in order to assess the company's financial stability and investment prospects.\n"
                     "UNANSWERABLE: The text only mention that a lawsuit occurred, but does not mention any specific consequences of the lawsuit\n"
                     "QUESTION: What potential financial damages or penalties is Anthropic facing as a result of the lawsuit filed by Universal Music and other plaintiffs\n\n"
                     "REASON (Answerable): Since the AI industry and Anthropic are both relatively new, retail investors may want to understand the regulatory and legal risks associated before making any move.\n"
                     "ANSWER: Anthropic is being sued by music publishers including Universal Music, Concord, and ABKCO over copyrighted lyrics generated by Claude.\n"
                     "QUESTION: What are some legal risks that Anthropic is currently facing?"
    }
]
SUMMARY_AUG_PARSE_SYSTEM = "You are a helpful AI assistant that will convert natural language into a machine parsable format. {format_instructions}"
SUMMARY_AUG_PARSE_EXAMPLE = [{"user": """REASON (Unanswerable): Retail investors would want to estimate the potential financial repercussions of the lawsuit on Anthropic in order to assess the company's financial stability and investment prospects.
UNANSWERABLE: The text only mention that a lawsuit occurred, but does not mention any specific consequences of the lawsuit
QUESTION: What potential financial damages or penalties is Anthropic facing as a result of the lawsuit filed by Universal Music and other plaintiffs?

REASON: Since the AI industry and Anthropic are both relatively new, retail investors may want to understand the regulatory and legal risks associated before making any move.
ANSWER: Anthropic is being sued by music publishers including Universal Music, Concord, and ABKCO over copyrighted lyrics generated by Claude.
QUESTION: What are some legal risks that Anthropic is currently facing?""",
                              "assistant": """{"answerable": "What are some legal risks that Anthropic is currently facing?", "unanswerable": "What potential financial damages or penalties is Anthropic facing as a result of the lawsuit filed by Universal Music and other plaintiffs?"}"""}]
SUMMARY_AUG_PARSE_USER = "{text}"
SUMMARY_AUG_MODEL = "chat-bison"
SUMMARY_AUG_TEMPERATURE = 0

# AI Assisted Summary Rewriting
OPENAI_KEY_PATH = "../key"
SUMMARY_REWRITE_SYSTEM = "You are a helpful AI assistant that will rewrite keypoints summary from a news article " \
                         "given a specific published date and article title " \
                         "so that it does not use " \
                         "any relative time such as 'yesterday', 'last week', 'this quarter' etc. In addition, " \
                         "ensure that the re-written keypoints can be read and understood clearly " \
                         "regardless of whether the title or published date is present. " \
                         "DO NOT add any information not conveyed in the original text. " \
                         "DO NOT add any information about specific dates unless it was in the original text. " \
                         "Follow the given examples and begin."
SUMMARY_REWRITE_EXAMPLES = [
    {
        "user": "Title: Microsoft beats earning and lead stock rallies\n"
                "Date: 2023-10-31, Tuesday\n"
                "* Stock rallies today morning as Microsoft beats earning from last quarter\n"
                "* New CEO Sneed joined Microsoft from Google\n"
                "* This year is the year of the AI revenue, expert says\n"
                "* Spokesperson also mentioned revenues from Q2 on Monday this week\n"
                "* Last week, the stock was impacted by lawsuits over AI copyright issues",
        "assistant": "* Stocks rallied on the morning of 2023-10-31 as Microsoft beats earning from 2023 Q3\n"
                     "* It was reported on 2023-10-31 that new CEO Sneed joined Microsoft from Google\n"
                     "* 2023 is the year of the AI revenue, expert says\n"
                     "* Spokesperson from Microsoft also mentioned revenues from 2023 Q2 on 2023-10-30\n"
                     "* On the week of 2023-10-22, Microsoft stock was impacted by lawsuits over AI copyright issues"
    },
    {
        "user": "Title: Oil exploration worker wage talks failed to reach a conclusion\n"
                "Date: 2023-05-24, Wednesday\n"
                "* Workers' wage talks break down\n"
                "* No strike allowed before mediation talks scheduled in June\n"
                "* Risk of strike later this year if mediation fails\n"
                "* Production workers separately agreed deal this month",
        "assistant": "* Oil exploration workers' wage talks break down, report on 2023-05-24 says\n"
                     "* No strike allowed for the workers before mediation talks scheduled in June, 2023\n"
                     "* Risk of strike later in 2023 if mediation fails\n"
                     "* Production workers separately agreed deal in May 2023"
    }
]
SUMMARY_REWRITE_USER = "Title: {title}\nDate: {date}\nText:\n{ipt_text}"

# AI Assisted Summary Filtering
SUMMARY_FILTER_SYSTEM = "You are a helpful AI assistant that will determine whether the given summary is relevant to a " \
                        "retail investor interested in different financial assets. Then, you will determine if the " \
                        "summary talks about the movement of share prices, index, or information that can easily " \
                        "be extracted from a stock screener. For each determination, " \
                        "first give the concise reasoning, then given the judgement. " \
                        "Follow the given examples and begin."
SUMMARY_FILTER_USER = "{summary}"
SUMMARY_FILTER_EXAMPLE = [
    {
        "user": "* Stock rallies today morning as Microsoft beats earning from last quarter\n"
                "* New CEO Sneed joined Microsoft from Google\n"
                "* This year is the year of the AI revenue, expert says\n"
                "* Spokesperson also mentioned revenues from Q2 on Monday this week\n"
                "* Last week, the stock was impacted by lawsuits over AI copyright issues",
        "assistant": "REASON FOR RELEVANCE TO INVESTOR: The summary mentions the increase in revenue from AI, "
                     "as well as potential risks such as copyright concerns over generated contents. In addition, "
                     "it talks about new CEO Sneed, who joined Microsoft from Google. Thus, the summary would be "
                     "relevant to an investor interested in investing in AI, and the change in CEO is a significant "
                     "event for Microsoft, which investors would look into.\nVERDICT FOR RELEVANT TO INVESTOR, True or False: True\n\n"
                     "REASON FOR WHETHER MOVEMENT MENTIONED: The summary talks about the stocks rallying today.\n"
                     "VERDICT FOR STATISTICS MENTIONED, True or False: True"
    },
    {
        "user": "* Oil exploration workers' wage talks break down\n" \
                "* No strike allowed before mediation talks scheduled in June\n" \
                "* Risk of strike later this year if mediation fails\n" \
                "* Production workers separately agreed deal this month",
        "assistant": "REASON FOR RELEVANCE TO INVESTOR: The summary mentions potential risk of strikes for the oil " \
                     "exploration workers, which may impact the oil and gas industry. " \
                     "Thus, investors in oil and gas industry would be interested to know about this risk" \
                     "VERDICT FOR RELEVANT TO INVESTOR, True or False: True\n\n" \
                     "REASON FOR WHETHER MOVEMENT MENTIONED: The summary does not mention any change in stock or index\n" \
                     "VERDICT FOR MOVEMENT MENTIONED, True or False: False"
    }
]

# Creating Summary Titles
SUMMARY_SUMMARY_SYSTEM = "You are a helpful AI assistant that will rewrite the given keypoints so that it includes a one line headline at the beginning. " \
                         "An example:\nKeypoints:\n" \
                         "* Universal Music has sued Anthropic, the AI startup, over “systematic and widespread infringement of their copyrighted song lyrics,” per a filing Wednesday in a Tennessee federal court.\n" \
                         "* Anthropic, the $4.1 billion startup behind Claude the chatbot, was founded in 2021 by former OpenAI research executives.\n" \
                         "* Other music publishers, such as Concord and ABKCO, were also named as plaintiffs.\n\n" \
                         "Response:\n" \
                         "Anthropic’s AI chatbot Claude is posting lyrics to popular songs, lawsuit claims"
SUMMARY_SUMMARY_USER = "{summary}"
