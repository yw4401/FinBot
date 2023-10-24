import datetime
import logging
import sys
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
    sub_sect = root.xpath('//meta[@name="article:section"]')[0]
    output["subsection"] = sub_sect.attrib["content"].strip()


@extractor_func(scraper="reuters", required=True)
def extract_article_title(root, output):
    header_tag = root.xpath('//meta[@property="og:title"]')[0]
    output["title"] = header_tag.attrib["content"].strip()


@extractor_func(scraper="reuters", required=True)
def extract_published_time(root, output):
    time_tag = root.xpath('//meta[@name="article:published_time"]')[0]
    output["published"] = time_tag.attrib["content"].strip()


@extractor_func(scraper="reuters", required=False)
def extract_summary(root, output):
    # Find all `div` elements that have a class containing "paragraph--"
    summary_tags = root.xpath('//ul[contains(@class, "summary__summary")]/li[contains(@data-testid, "Body")]')

    summary = ''
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

    body = ''
    # Extract the text content of each `div` element and print it
    for paragraph_tag in paragraph_tags:
        text = paragraph_tag.text_content().strip()
        if text:
            body = body + '\n\n' + text

    output["body"] = body


@consolidate_func(scraper="reuters")
def convert_reuters(reuter_dict):
    title = reuter_dict["title"]
    category = reuter_dict["subsection"]
    if len(title) < len(category):
        title, category = category, title
    format_string = "%Y-%m-%dT%H:%M:%SZ"
    published = datetime.datetime.strptime(reuter_dict["published"].strip(), format_string)
    published = published.replace(tzinfo=datetime.timezone.utc)

    body = ""
    for b in reuter_dict["body"].split("\n\n"):
        body = body + normalize_text(b) + "\n\n"
    body = body.strip()

    summary = ""
    summary_type = SummaryType.NULL
    if "summary" in reuter_dict:
        sum_list = reuter_dict["summary"].split("\n\n")
        summary, summary_type = consolidate_summary(sum_list)

    return Article(
        source=reuter_dict["source"],
        html=reuter_dict["html"],
        url=reuter_dict["url"],
        category=category,
        title=title,
        published=published,
        body=body,
        summary=summary,
        summary_type=summary_type.name
    )


def create_new_state(credential_path=None):
    writer = BigQueryWriter(project=GCP_PROJECT)
    with writer:
        tracker = InMemProgressTracker(starting_set=[BASE_URL],
                                       visited=writer.saved_pages,
                                       filters=[create_robot_filter(BASE_URL),
                                                create_regex_filter(r"https?://www\.reuters\.com.*")])
    getter = RequestGetter(retry=3)
    return writer, tracker, getter


RESET_TIME = 3600 * 24

if __name__ == "__main__":
    log_client = google.cloud.logging.Client(project="msca310019-capstone-f945")
    log_client.setup_logging()
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

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
