import os
import scrapy
import json
import oss2
from typing import Any
from enum import Enum
from urllib.parse import unquote
from lxml import html
from pydantic import BaseModel
from datetime import datetime

from g_crawler.etree_tools import clean_html, apply_trim_rules, to_string
from g_crawler.md_converter import GMarkdownConverter
from g_crawler.n_gram_tools import deduplicate_text

CONFIG_ROOT = "configs"


class Item(BaseModel):
    url: str
    title: str
    modified_at: datetime
    markdown: str
    html: str
    save_to: str


class OssWriterPipeline:
    """
    write item to oss, with meta `modified-at`, `title`

    the oss url is:
    * oss://raw_html/{item.save_to}
    * oss://markdown/{item.save_to}
    """

    def __init__(self) -> None:

        auth = oss2.Auth(
            os.getenv("AWS_ACCESS_KEY_ID"),
            os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
        self.bucket = oss2.Bucket(
            auth,
            os.getenv("AWS_ENDPOINT"),
            os.getenv("DOCUMENTS_BUCKET"),
        )

    def process_item(self, item, spider):
        item = Item(**item)

        # 获取元数据
        try:
            head_object_result = self.bucket.head_object(f"markdown/{item.save_to}.md")
            if "x-oss-meta-modified-at" in head_object_result.headers:
                last_modified_at = head_object_result.headers["x-oss-meta-modified-at"]
                if last_modified_at == str(item.modified_at):
                    return
        except oss2.exceptions.NoSuchKey:
            pass

        headers = {
            "x-oss-meta-modified-at": str(item.modified_at),
            "x-oss-meta-title": item.title.encode("utf-8"),
        }

        # 写入markdown
        self.bucket.put_object(
            f"markdown/{item.save_to}.md",
            item.markdown.encode("utf-8"),
            headers=headers,
        )

        # 写入html
        self.bucket.put_object(
            f"raw_html/{item.save_to}.html",
            item.html.encode("utf-8"),
            headers=headers,
        )


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
            samples = self._config["samples"]
            for sample in samples:
                url = f"{BWikiSpider.host}/{site_name}/{sample}"
                yield scrapy.Request(url, meta=meta)
        else:
            home_url = f"{BWikiSpider.host}/{site_name}/{BWikiSpider.HOME_PAGE}"
            yield scrapy.Request(home_url, meta=meta)

    def parse(self, response, **kwargs):
        # get meta from response
        title = response.xpath("//head/meta[@property='og:title']/@content").get()
        modified_at = response.xpath(
            "//head/meta[@property='article:modified_time']/@content"
        ).get()
        raw_html = response.css(self.MAIN_SELECTOR).get()

        etree = html.fragment_fromstring(raw_html)
        etree = clean_html(etree)
        etree = apply_trim_rules(etree, self._config["trim_rules"])
        cleaned_html = to_string(etree)

        # convert to md
        markdown = self._converter.convert(cleaned_html)
        markdown = deduplicate_text(markdown)

        url = unquote(response.url, encoding="utf-8", errors="replace")
        save_to = url.split("://")[-1]
        item = Item(
            save_to=save_to,
            url=url,
            title=title,
            modified_at=modified_at,
            markdown=markdown,
            html=raw_html,
        )

        if self._mode == CrawlingModes.SAMPLED:
            return item.model_dump(mode="json")

        # TODO

    # def get_meta
