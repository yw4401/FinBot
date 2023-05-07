from unstructured.partition.html import partition_html
from lxml.html import tostring
from unstructured.staging.base import convert_to_dict
from nltk.tokenize import word_tokenize
from common import *

BASE_URL = "https://www.cnbc.com"
LINKS_XPATH = "//a"


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


if __name__ == "__main__":
    start_scraper("cnbc",
                  progressor=InMemProgressTracker(starting_set=[BASE_URL],
                                                  filters=[create_robot_filter(BASE_URL),
                                                           create_regex_filter(r"https?://www\.cnbc\.com")]),
                  writer=JSONFileDirectoryWriter("../cnbc-scrape"),
                  delay=1)
