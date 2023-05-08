import logging
import time

from unstructured.partition.html import partition_html
from lxml.html import tostring
from unstructured.staging.base import convert_to_dict
from nltk.tokenize import word_tokenize
from common import *
import google.cloud.logging
import google.auth


BASE_URL = "https://www.reuters.com/"
LINKS_XPATH = "//a"


@extractor_func(scraper="reuters", required=True)
def extract_article_section(root, output):
    sub_sect = root.xpath("//h1[@data-testid='Heading']")[0]
    output["subsection"] = sub_sect.text_content().strip()

@extractor_func(scraper="reuters", required=True)
def extract_article_title(root, output):
    header_tag = root.xpath('//span[contains(@class,"heading__heading_4")]')[0]
    output["title"] = header_tag.text_content().strip()


@extractor_func(scraper="reuters", required=True)
def extract_published_time(root, output):
    # Find the `time` element with the class 'article-header__dateline__4jE04'
    time_tag = root.xpath('//time[contains(@class, "article-header__dateline")]')[0]

    # Extract the text content of each 'span' element within the 'time' element
    for span_tag in time_tag.xpath('./span'):
        if 'Last Updated' not in span_tag.text_content():
            if 'AM' in span_tag.text_content() or 'PM' in span_tag.text_content():
                time_text = span_tag.text_content().strip().replace('UTC', '')
            else:
                date_text = span_tag.text_content().strip()
 
    # Concatenate date and time information and print
    datetime_str = date_text + ' ' + time_text

    output["published"] = datetime_str


@extractor_func(scraper="reuters", required=False)
def extract_summary(root, output):
    # Find all `div` elements that have a class containing "paragraph--"
    summary_tags = root.xpath('//ul[contains(@class, "summary__summary")]/li[contains(@data-testid, "Body")]')
    
    summary=''
    # Extract the text content of each `div` element and print it
    for summary_tag in summary_tags:
        text = summary_tag.text_content().strip()
        if text:
            summary = summary + '\n\n' + text

    output["summary"] = summary


@extractor_func(scraper="reuters", required=True)
def extract_body(root, output):
    # Find all `div` elements that have a class containing "paragraph--"
    paragraph_tags = root.xpath('//div/p[contains(@data-testid, "paragraph-")]')
    
    body=''
    # Extract the text content of each `div` element and print it
    for paragraph_tag in paragraph_tags:
        text = paragraph_tag.text_content().strip()
        if text:
            body = body + '\n\n' + text

    output["body"] = body
  


def create_new_state(credential_path=None):
    writer = GCPBucketDirectoryWriter(bucket="reuters-articles",
                                      credential_path=credential_path
                                      )
    with writer:
        tracker = InMemProgressTracker(starting_set=[BASE_URL],
                                       visited=writer.saved_pages,
                                       filters=[create_robot_filter(BASE_URL),
                                                create_regex_filter(r"https?://www\.reuters\.com.*")])
    getter = RequestGetter(retry=3)
    return writer, tracker, getter


RESET_TIME = 3600 * 24


if __name__ == "__main__":
    #credentials = "/home/sdai/.config/gcloud/application_default_credentials.json"
    #auto_credentials, project_id = google.auth.default()

    log_client = google.cloud.logging.Client(project="msca310019-capstone-f945")
    log_client.setup_logging()
    logging.getLogger().setLevel(logging.INFO)

    writer, tracker, getter = create_new_state()
    last_reset = time.time()
    while True:
        current_time = time.time()
        if current_time - last_reset > RESET_TIME:
            writer, tracker, getter = create_new_state()
            logging.info("Restarting scrape to get new articles")
            last_reset = time.time()
        start_scraper("reuters", getter=getter, writer=writer, progressor=tracker, duration=5 * 60)
        with writer:
            writer.write_index()
