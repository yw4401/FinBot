from selenium.common.exceptions import WebDriverException, NoSuchElementException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from selenium import webdriver
from collections import namedtuple
from fake_useragent import UserAgent
from selenium.webdriver.chrome.service import Service
import locale
import json
import time
import re

ua = UserAgent()

BASE_URL = "https://www.cnbc.com"
HEADERS = {
    "User-Agent": ua.random
}

chrom_param = "--lang=en,--start-maximized,--ignore-certificate-errors"

LINKS_XPATH = "//a"
PAGE_LOAD = "//a[@class='branding-menu-logo']"


def normalize_url(url):
    if url[-1] == "#" or url[-1] == "/":
        url = url[:-1]
    return url


def download_pages(web_driver):
    link_objs = web_driver.find_elements(By.XPATH, LINKS_XPATH)
    result_set = set()
    for l in link_objs:
        if l.get_attribute("href"):
            url = normalize_url(l.get_attribute("href"))
            result_set.add(url)
    return result_set, web_driver.page_source


def get_urls(web_driver, start_url, allowed_fqdn=".*", emit=lambda driver: True):
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
        web_driver.get(next_url)
        page_links, source = download_pages(web_driver)
        queue.extend(page_links)
        visited.add(next_url)
        if emit(web_driver):
            yield next_url, source


if __name__ == "__main__":
    chrome_options = Options()
    for param in chrom_param.split(","):
        if len(param.strip()) != 0:
            chrome_options.add_argument(param.strip())
    driver = webdriver.Chrome(service=Service(r"../chromedriver"), options=chrome_options)
    wait = WebDriverWait(driver, 3600)

    counter = 0
    for url, source in get_urls(driver, BASE_URL, allowed_fqdn=r".*cnbc\.com.*"):
        with open("../cnbc_scrape/%s.json" % counter, "w") as fp:
            json.dump({"url": url, "source": source}, fp)
    counter = counter + 1
