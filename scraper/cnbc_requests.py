import requests
import json
from unstructured.partition.html import partition_html
import re
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from lxml.html import fromstring
from lxml.html import tostring
from unstructured.staging.base import convert_to_dict
from nltk.tokenize import word_tokenize


ua = UserAgent()
HEADERS = {
    "User-Agent": ua.random,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,"
              "application/signed-exchange;v=b3;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9"
}


BASE_URL = "https://www.cnbc.com"
LINKS_XPATH = "//a"


def normalize_url(url, source_url):
    url = urljoin(source_url, url)
    if url[-1] == "#" or url[-1] == "/":
        url = url[:-1]
    return url


def extract_page(resp, source_url):
    root = BeautifulSoup(resp.content, "html.parser")
    result_set = set()
    for l in root.find_all("a", href=True):
        url = normalize_url(l["href"], source_url)
        result_set.add(url)
    return result_set, root


def get_urls(start_url, allowed_fqdn=".*", emit=lambda root: True):
    visited = set()
    url_matcher = re.compile(allowed_fqdn)
    queue = [start_url]
    while len(queue) != 0:
        next_url = queue.pop(0)
        if next_url in visited:
            continue
        if not url_matcher.match(next_url):
            continue
        print("Visiting: " + next_url)
        visited.add(next_url)
        resp = requests.get(url=next_url, headers=HEADERS)
        try:
            page_links, root = extract_page(resp, next_url)
        except ValueError:
            print("Failed to get: " + next_url)
            continue
        queue.extend(page_links)
        if emit(root):
            print("Emitting: " + next_url)
            yield next_url, str(root)
#        time.sleep(1)


def find_headline_emitter(root):
    headlines = root.find_all("h1", class_="ArticleHeader-headline")
    subsections = root.find_all("a", class_="ArticleHeader-eyebrow")
    return len(headlines) != 0 and len(subsections) != 0


def extract_article_section(root, output):
    sub_sect = root.xpath("//a[@class='ArticleHeader-eyebrow']")[0]
    output["subsection"] = sub_sect.text_content().strip()


def extract_article_title(root, output):
    header_tag = root.xpath("//h1[@class='ArticleHeader-headline']")[0]
    output["title"] = header_tag.text_content().strip()


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


extractors = [extract_article_section, extract_article_title, extract_summary, extract_body]


if __name__ == "__main__":
    counter = 0
    for url, source in get_urls(BASE_URL, allowed_fqdn=r".*www\.cnbc\.com.*", emit=find_headline_emitter):
        root = fromstring(source)
        with open("../cnbc_scrape/%s.json" % counter, "w") as fp:
            output_dict = {"url": url, "source": source}
            for ext in extractors:
                ext(root, output_dict)
            json.dump(output_dict, fp)
        counter = counter + 1
