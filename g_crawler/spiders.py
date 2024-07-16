import scrapy
import json
from typing import Any
from enum import Enum
from urllib.parse import unquote
from lxml import html

from g_crawler.etree_tools import clean_html, apply_trim_rules, to_string
from g_crawler.md_converter import GMarkdownConverter
from g_crawler.n_gram_tools import deduplicate_text

CONFIG_ROOT = "configs"


class CrawlingModes(Enum):
    FULL = "full"
    SAMPLED = "sampled"


class BWikiSpider(scrapy.Spider):
    name = "BWiki"
    host = "https://wiki.biligame.com"
    HOME_PAGE = "首页"
    MAIN_SELECTOR = ".mw-parser-output"

    def __init__(
        self,
        name: str,  # name is the site name
        mode: CrawlingModes = CrawlingModes.FULL,
        **kwargs: Any,
    ):
        super().__init__(f"{BWikiSpider.name}/{name}", **kwargs)

        # load config file
        config_file_path = f"{CONFIG_ROOT}/{name}/meta.json"
        with open(config_file_path, "r") as f:
            self._config = json.load(f)

        self._mode = mode
        self._converter = GMarkdownConverter()

    def start_requests(self):
        meta = {"playwright": True}
        site_name = self.name.split("/")[-1]

        if self._mode == CrawlingModes.SAMPLED:
            # samples = self._config["samples"]
            # for sample in samples:
            #     url = f"{BWikiSpider.host}/{site_name}/{sample}"
            #     yield scrapy.Request(url, meta=meta)
            yield scrapy.Request(
                "https://wiki.biligame.com/ys/%E5%B8%8C%E6%A0%BC%E9%9B%AF",
                meta=meta,
            )
        else:
            home_url = f"{BWikiSpider.host}/{site_name}/{BWikiSpider.HOME_PAGE}"
            yield scrapy.Request(home_url, meta=meta)

    def parse(self, response, **kwargs):
        # get meta from response
        title = response.xpath("//head/meta[@property='og:title']/@content").get()
        modified_at = response.xpath(
            "//head/meta[@property='article:modified_time']/@content"
        ).get()
        content = response.css(self.MAIN_SELECTOR).get()

        etree = html.fragment_fromstring(content)
        etree = clean_html(etree)
        etree = apply_trim_rules(etree, self._config["trim_rules"])
        content = to_string(etree)

        # convert to md
        markdown = self._converter.convert(content)
        markdown = deduplicate_text(markdown)

        if self._mode == CrawlingModes.SAMPLED:
            return {
                "url": unquote(response.url, encoding="utf-8", errors="replace"),
                "title": title,
                "modified_at": modified_at,
                "content": markdown,
            }

        # TODO

    # def get_meta
