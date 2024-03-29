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


BASE_URL = "https://www.cnbc.com"
LINKS_XPATH = "//a"
ARTICLE_CONVERT_CNBC_DATE = "%Y-%m-%dT%H:%M:%S%z"


@extractor_func(scraper="cnbc", required=True)
def extract_article_section(root, output):
    sub_sect = root.xpath("//a[@class='ArticleHeader-eyebrow']")[0]
    output["subsection"] = sub_sect.text_content().strip()


@extractor_func(scraper="cnbc", required=True)
def extract_article_title(root, output):
    header_tag = root.xpath("//h1[@class='ArticleHeader-headline']")[0]
    output["title"] = header_tag.text_content().strip()


@extractor_func(scraper="cnbc", required=True)
def extract_published_time(root, output):
    time_tag = root.xpath("//time[@data-testid='published-timestamp']")[0]
    output["published"] = time_tag.attrib["datetime"]


@extractor_func(scraper="cnbc", required=False)
def extract_summary(root, output):
    key_point_list = root.xpath("//div[@class='RenderKeyPoints-list']//ul")
    if len(key_point_list) > 0:
        key_point_list = key_point_list[0]
    else:
        return
    key_points = []
    for p in key_point_list.xpath(".//li"):
        key_points.append(p.text_content().strip())
    output["summary"] = key_points


@extractor_func(scraper="cnbc", required=True)
def extract_body(root, output):
    body_section = partition_html(text=tostring(root.xpath("//div[contains(@class, 'ArticleBody-articleBody')]")[0]))
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


@consolidate_func(scraper="cnbc")
def convert_cnbc(cnbc_dict):
    title = cnbc_dict["title"]
    category = cnbc_dict["subsection"]
    published = datetime.datetime.strptime(cnbc_dict["published"], ARTICLE_CONVERT_CNBC_DATE)

    body = ""
    for d in cnbc_dict["body"]:
        if d["type"] == "Title":
            body = body + "## " + normalize_text(d["text"]) + "\n\n"
        if d["type"] == "NarrativeText":
            body = body + normalize_text(d["text"]) + "\n\n"
    body = body.strip()

    summary = ""
    summary_type = SummaryType.NULL
    if "summary" in cnbc_dict:
        sum_list = cnbc_dict["summary"]
        summary, summary_type = consolidate_summary(sum_list)
    return Article(
        source=cnbc_dict["source"],
        html=cnbc_dict["html"],
        url=cnbc_dict["url"],
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
                                                create_regex_filter(r"https?://www\.cnbc\.com")])
    getter = RequestGetter(retry=3)
    return writer, tracker, getter


RESET_TIME = 3600 * 24


if __name__ == "__main__":
    log_client = google.cloud.logging.Client(project=GCP_PROJECT)
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
        start_scraper("cnbc", getter=getter, writer=writer, progressor=tracker, duration=5 * 60)
        with writer:
            writer.write_index()
