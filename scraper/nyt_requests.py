import logging
import time

from unstructured.partition.html import partition_html
from lxml.html import tostring
from unstructured.staging.base import convert_to_dict
from nltk.tokenize import word_tokenize
from common import *
import google.cloud.logging
import google.auth


BASE_URL = "https://www.nytimes.com"
LINKS_XPATH = "//a"


@extractor_func(scraper="nyt", required=True)
def extract_article_section(root, output):
    sub_sect = root.xpath("//div[@id='masthead-section-label']")[0]
    output["subsection"] = sub_sect.text_content().strip()


@extractor_func(scraper="nyt", required=True)
def extract_article_title(root, output):
    header_tag = root.xpath("//h1[@data-testid='headline']")[0]
    output["title"] = header_tag.text_content().strip()


@extractor_func(scraper="nyt", required=True)
def extract_published_time(root, output):
    time_tag = root.xpath("//time[@class='css-1g7pp1u e16638kd0']")[0]
    output["published"] = time_tag.attrib["datetime"]


@extractor_func(scraper="nyt", required=False)
def extract_summary(root, output):
    key_points = root.xpath("//p[@id='article-summary']")[0]
    output["summary"] = key_points.text_content().strip()


@extractor_func(scraper="nyt", required=True)
def extract_body(root, output):
    body_section = partition_html(text=tostring(root.xpath("//section[@name='articleBody']")[0]))
    body_dicts = convert_to_dict(body_section)
    accepted = {"Title", "NarrativeText"}
    result = []

    for d in body_dicts:
        if len(word_tokenize(d["text"])) < 3:
            continue
        if "|" in d["text"]:
            continue
        if d["type"] not in accepted:
            continue
        result.append(
            {
                "type": d["type"],
                "text": d["text"]
            }
        )

    output["body"] = result


def create_new_state(credential_path=None):
    writer = GCPBucketDirectoryWriter(bucket="nyt-articles",
                                      credential_path=credential_path
                                      )
    with writer:
        tracker = InMemProgressTracker(starting_set=[BASE_URL],
                                       visited=writer.saved_pages,
                                       filters=[create_robot_filter(BASE_URL),
                                                create_regex_filter(r"https?://www\.nytimes\.com")])
    getter = RequestGetter(retry=3)
    return writer, tracker, getter


RESET_TIME = 3600 * 24


if __name__ == "__main__":
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
        start_scraper("nyt", getter=getter, writer=writer, progressor=tracker, duration=5 * 60)
        with writer:
            writer.write_index()
