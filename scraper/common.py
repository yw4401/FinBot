import time
import json
import requests
import re
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from lxml.html import fromstring
from contextlib import AbstractContextManager
import os
from pathlib import Path
from types import TracebackType
from typing import Type
from urllib.parse import urlsplit
import urllib.robotparser
from random import randint


ua = UserAgent()
registered_extractors = {}


LINKS_XPATH = "//a"


def extractor_func(scraper, required=False):
    """
    Registers an extractor for a given scraper. The extractor will be used to fill the output dictionary when
    the start_scraper function is called.
    :param scraper: the name of the scraper
    :param required: whether the page should still be processed when this extractor fails
    :return: the registered extractor function
    """
    if scraper not in registered_extractors:
        registered_extractors[scraper] = []

    def extractor_decorator(func):
        registered_extractors[scraper].append({
            "required": required, "method": func
        })
        return func

    return extractor_decorator


def start_scraper(scraper, progressor, writer, getter=None, delay=1, duration=float("inf")):
    """
    Start to scrape the webpages from the progressor using the extractors from the scraper, and write the extracted
    output dictionary via the writer. In the process, the getter is used to download the page and get all the link
    with the source of the page. Between each download, there will be a wait of approx. delay seconds.
    :param scraper: the name that corresponds to the registered extractors
    :param progressor: an iterable object that implements an update function, which should take an iterable of urls. The
    role of the progressor is to provide the next url to scrape, and keep track of the page scraped thus far.
    :param writer: a context manager that supports a write function, which takes the extracted output dictionary and saves it
    :param getter: an object that supports an extract function. The extract function should take an url, and return a tuple
    with the first element being an iterable of links from the page identified by the url, and the second element should be
    a string containing the source of the page in html
    :param delay: the number of seconds to pause between each visit to the site
    :param duration: the approximate amount of seconds to run the scraper
    """

    if getter is None:
        getter = RequestGetter()
    with writer:
        start_time = time.time()
        for url, source in get_urls(progressor=progressor, getter=getter, delay=delay):
            root = fromstring(source)
            extractors = []
            if scraper in registered_extractors:
                extractors = registered_extractors[scraper]
            output_dict = {"url": url, "source": source}
            write = True
            for ext in extractors:
                required = ext["required"]
                try:
                    func = ext["method"]
                    func(root, output_dict)
                except Exception as e:
                    print("Failed to Extract: " + str(e))
                    if required:
                        write = False
                    break
            if write:
                writer.save(output_dict)
            if time.time() - start_time >= duration:
                break


def get_urls(progressor, getter=None, delay=1):
    """
    Continuously visits the next page as directed by the progressor, and yields the url visited as well as the source of
    the page identified by the url.
    :param progressor: an iterable object that implements an update function, which should take an iterable of urls. The
    role of the progressor is to provide the next url to scrape, and keep track of the page scraped thus far.
    :param getter: an object that supports an extract function. The extract function should take an url, and return a tuple
    with the first element being an iterable of links from the page identified by the url, and the second element should be
    a string containing the source of the page in html
    :param delay: the number of seconds to pause between each visit to the site
    :return: a generator that will yield a tuple where the first element is the url of the page visited, and the second
    element is the source of the page
    """

    if getter is None:
        getter = RequestGetter()
    for next_url in progressor:
        print("Visiting: " + next_url)
        try:
            page_links, source = getter.extract(next_url)
            progressor.update(page_links)
            yield next_url, source
        except Exception as e:
            print("Failed to extract page: " + str(e))
        time.sleep(delay)


def normalize_url(url, source_url):
    """
    Convert potentially relative urls to absolute urls, and remove any slash or hash at the end
    :param url: the url to normalize
    :param source_url: the url of the page that the url to be normalized was found on
    :return: an absolute url without extra hash or slash at the end
    """
    url = urljoin(source_url, url)
    if url[-1] == "#" or url[-1] == "/":
        url = url[:-1]
    return url


def get_links(root, base_url):
    """
    Get all html links from a beautiful soup object
    :param root: the root node of the bs4 BeautifulSoup object
    :param base_url: the source url of the page
    :return: all links on the page as absolute urls, without extra slash or hash at the end
    """
    result_set = set()
    for l in root.find_all("a", href=True):
        url = normalize_url(l["href"], base_url)
        result_set.add(url)
    return result_set


class RequestGetter:
    """
    A class that is used by the scraper to extract urls and page sources using the requests library
    """

    def __init__(self, timeout=30, retry=10, headers=None, proxies=()):
        """
        initializes the RequestGetter object with http request related parameters.
        :param timeout: the maximum amount of time to wait for server to respond in seconds
        :param retry: the maximum number of times to retry to failed to get a proper response
        :param headers: a dictionary of headers that will be added to the request in addition to the default headers
        :param proxies: a list of urls to proxies that will be used by the :func:`common.RequestGetter.extract`
        """
        if headers is None:
            self.headers = {}
        self.timeout = timeout
        self.retry = retry
        self.proxy = list(proxies)

    def generate_header(self):
        """
        generates the header that will be used by the extract function. It will create a random user agent with
        some common headers sent by a regular browsers. In addition, it will also add the custom header fields.
        :return: the header dictionary
        """
        header = {
            "User-Agent": ua.random,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,"
                      "application/signed-exchange;v=b3;q=0.7",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "en-US,en;q=0.9"
        }
        header.update(self.headers)
        return header

    def get_random_proxy(self):
        """
        selects a random proxy from the list of given proxy, and put it in an appropriate format for the requests library.
        :return: a randomly selected proxy or None if no proxy is given
        """
        if len(self.proxy) == 0:
            return {}
        idx = randint(0, len(self.proxy) - 1)
        return {
            "http": self.proxy[idx],
            "https": self.proxy[idx]
        }

    def extract(self, next_url):
        """
        Attempts to get the page source and all links in absolute url form from the given url. It will use the specified
        timeout, number of retry, as well as a randomly picked proxy from the given list of proxies. Furthermore, the
        specified header parameters in the constructor will be passed as part of the http request.
        :param next_url: the url of the page to download and extract
        :return: a tuple with the first element being the links from the page in absolute format, and the second element
        being the source of the page
        """
        retry_count = 0
        resp = None
        while retry_count < self.retry:
            try:
                rand_proxy = self.get_random_proxy()
                resp = requests.get(url=next_url, headers=self.generate_header(),
                                    timeout=self.timeout, proxies=rand_proxy)

                # Set the resp variable to the actual response if and only if a proper response is returned from the
                # server.
                old_resp = resp
                resp = None
                old_resp.raise_for_status()
                resp = old_resp
                break
            except Exception as e:
                print("Failed to fetch: " + next_url)
                print("Exception: " + str(e))
                if retry_count != self.retry:
                    print("Retrying")
                retry_count = retry_count + 1
        if resp is None:
            raise ValueError("Failed to fetch: " + next_url)
        root = BeautifulSoup(resp.content, "html.parser")
        return get_links(root, next_url), str(root)


class JSONFileDirectoryWriter(AbstractContextManager):

    def __init__(self, directory):
        AbstractContextManager.__init__(self)
        self.directory = directory
        self.counter = 0

    def save(self, output):
        if not os.path.exists(self.directory):
            Path(self.directory).mkdir(parents=True)
        if not os.path.isdir(self.directory):
            raise ValueError("Invalid Directory: " + self.directory)
        with open(Path(self.directory, str(self.counter) + ".json"), "w") as fp:
            json.dump(output, fp)
            self.counter = self.counter + 1

    def __exit__(self, __exc_type: Type[BaseException] | None, __exc_value: BaseException | None,
                 __traceback: TracebackType | None) -> bool | None:
        pass


class InMemProgressTracker:

    def __init__(self, starting_set, visited=(), filters=()):
        self.queue = list(set(starting_set))
        self.visited = set(visited)
        self.filters = list(filters)

    def update(self, links):
        filtered_links = set()
        for l in links:
            filtered = False
            for f in self.filters:
                if not f(l):
                    filtered = True
                    break
            if not filtered:
                filtered_links.add(l)
        for l in filtered_links:
            if l not in self.visited:
                self.queue.append(l)

    def __iter__(self):
        return self

    def __next__(self):
        while len(self.queue) != 0:
            candidate = self.queue.pop(0)
            if candidate not in self.visited:
                self.visited.add(candidate)
                return candidate
        raise StopIteration()


def parse_robot_txt(base_url):
    url_parts = urlsplit(base_url)
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url("%s://%s/robots.txt" % (url_parts.scheme, url_parts.netloc))
    rp.read()
    return rp


def create_robot_filter(base_url):
    robot = parse_robot_txt(base_url)

    def pruner(url):
        return robot.can_fetch("*", url)

    return pruner


def create_regex_filter(regex):

    pattern = re.compile(regex)

    def pruner(url):
        return pattern.match(url)

    return pruner



