# FinBot

## Project Description
The Retail Investor News Analysis web application uses large language models and Natural Language Processing (NLP) to analyze news articles and provide insights to retail investors. The application allows users to stay informed about market trends, sentiments, and important events by analyzing a specified time period of news articles.

## Project Overflow

### Web Scraping for Data Collection ([scraper](scraper))
- We've gathered data from prominent news sources, including CNBC, The New York Times, and Reuters, as part of the data collection phase. The [common.py](scraper/common.py) is a framework that serves as the backbone for the web scraper. It encapsulates the essential functionality of traversing HTML pages by following links, verifying if a URL has already been scraped, and progressing to the subsequent page.
- The [cnbc_requests.py](scraper/cnbc_requests.py), [nyt_requests.py](scraper/nyt_requests.py) and [reuters_requests.py](scraper/reuters_requests.py) scripts are dedicated scrapers designed for their respective news sources, employing the '[lxml](https://lxml.de/)' module to extract key information such as article sections, titles, publication timestamps, article bodies, and optional summaries.
- News articles spanning from March 2008 to September 2023 have been scraped and collected for the project.

### Consolidating all the news articles
- The [consolidate_scraper_1.py](pipeline/consolidate_scraper_1.py) script serves as the central hub for aggregating and formatting news articles gathered from various sources.

### Monthly Data Segregation
- The [select_data_2.py](pipeline/select_data_2.py) script partitions the data into separate segments, dividing it on a monthly basis. These segments are then stored in distinct Parquet files.

### Deduplication of the news articles
- The dataset includes duplicate articles, which may be attributed to either the scraper's behavior or the occurrence of similar news articles from different sources.
-  Eliminating these duplicate news articles is crucial to improve efficiency, prevent potential biases caused by redundant information, and ensure a more accurate representation of the data.
-  In this context, we've employed the MinHash LSH technique to effectively eliminate duplicate news articles from the dataset.

### Coreference Resolution for Text Simplification
- The [coref_resolve_4.py](pipeline/coref_resolve_4.py) script utilizes a list of coreference clusters to transform a Spacy document into a string. In this transformation, each coreference is replaced with its primary mention, enhancing the clarity and coherence of the text.

### Topic Modelling
- The primary objective of topic modeling ([extract_topics_5.py](pipeline/extract_topics_5.py)) is to ensure the retrieval of a diverse set of content segments. Typically, in semantic search, there's a possibility that the top segments retrieved could be quite similar. However, by introducing topics, it compels the system to select segments that cover distinct facets or angles of the subject matter, promoting a more comprehensive and well-rounded result set.
- Here, the topic modeling is done using BERTopic.

### Topic Summarizer
- The [summarize_topics_6.py](pipeline/summarize_topics_6.py) produces a summary for each topic generated above.

### Finetuning the data
- ....... 

