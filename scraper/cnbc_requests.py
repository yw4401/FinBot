import requests
import json
import time
import re
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
from urllib.parse import urljoin


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
        visited.add(next_url)
        print("Visiting: " + next_url)
        resp = requests.get(url=next_url, headers=HEADERS)
        try:
            page_links, root = extract_page(resp, next_url)
        except ValueError:
            print("Failed to get: " + next_url)
            continue
        queue.extend(page_links)
        if emit(root):
            yield next_url, str(root)
        time.sleep(1)


if __name__ == "__main__":
    counter = 0
    for url, source in get_urls(BASE_URL, allowed_fqdn=r".*cnbc\.com.*"):
        with open("../cnbc_scrape/%s.json" % counter, "w") as fp:
            json.dump({"url": url, "source": source}, fp)
        counter = counter + 1
