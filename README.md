# FinBot

## Project Description
The Retail Investor News Analysis web application uses large language models and Natural Language Processing (NLP) to analyze news articles and provide insights to retail investors. The application allows users to stay informed about market trends, sentiments, and important events by analyzing a specified time period of news articles.

## Project Overflow

### Web Scraping for Data Collection [scraper](scraper)
- We've gathered data from prominent news sources, including CNBC, The New York Times, and Reuters, as part of the data collection phase. The [common.py](scraper/common.py) is a framework that serves as the backbone for the web scraper. It encapsulates the essential functionality of traversing HTML pages by following links, verifying if a URL has already been scraped, and progressing to the subsequent page.
- The [cnbc_requests.py](scraper/cnbc_requests.py), [nyt_requests.py](scraper/nyt_requests.py) and [reuters_requests.py](scraper/reuters_requests.py) scripts are dedicated scrapers designed for their respective news sources, employing the '[lxml](https://lxml.de/)' module to extract key information such as article sections, titles, publication timestamps, article bodies, and optional summaries.
- News articles spanning from March 2008 to September 2023 have been scraped and collected for the project.


