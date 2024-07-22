import os
import re
import scrapy
import json
import oss2
from typing import Any
from enum import Enum
from urllib.parse import unquote
from lxml import html
from pydantic import BaseModel
from datetime import datetime
from scrapy.linkextractors import LinkExtractor
from urllib.parse import urlparse
from tqdm import tqdm

from g_crawler.etree_tools import clean_html, apply_trim_rules, to_string
from g_crawler.md_converter import GMarkdownConverter
from g_crawler.n_gram_tools import deduplicate_text


CONFIG_ROOT = "configs"
IGNORED_PATH_PREFIX = [
    "文件:",
    "模板:",
    "用户:",
    "特殊:",
    "MediaWiki:",
    "File:",
    "Widget:",
    "User:",
    "File:",
    "Data:",
    "分类:",
]


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

    def _put_item_to_oss(self, item: Item):
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

    def process_item(self, item, spider):
        item = Item(**item)
        if any([prefix in item.save_to for prefix in IGNORED_PATH_PREFIX]):
            return item

        try:
            self._put_item_to_oss(item)
        except oss2.exceptions.RequestError:
            spider.logger.warning(f"Failed to write item to oss: {item.url}, skip.")

        return item


class MarkdownWriterPipeline:
    def process_item(self, item, spider):
        item = Item(**item)
        if any([prefix in item.save_to for prefix in IGNORED_PATH_PREFIX]):
            return item

        # write to markdown
        save_to = f"{os.getenv('CRAWLER_MARKDOWN_ROOT', 'outputs')}/{item.save_to}.md"
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        with open(save_to, "w") as f:
            f.write(item.markdown)

        return item


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

        self._site_name = name
        self._mode = mode
        self._converter = GMarkdownConverter()
        self._link_extractor = LinkExtractor(unique=True, strip=True)
        self._processed_links = set()
        self._progress_bar = tqdm(desc=f"Crawling {self._site_name}", unit=" page")

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
        self._processed_links.add(response.url)

        # get meta from response
        title = (
            response.xpath("//head/meta[@property='og:title']/@content").get().strip()
        )
        modified_at = response.xpath(
            "//head/meta[@property='article:modified_time']/@content"
        ).get()
        raw_html = response.css(self.MAIN_SELECTOR).get()
        if not raw_html:
            return

        etree = html.fragment_fromstring(raw_html)
        etree = clean_html(etree)
        etree = apply_trim_rules(etree, self._config.get("trim_rules", []))
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
        if title != BWikiSpider.HOME_PAGE:
            self._progress_bar.update(1)
            yield item.model_dump(mode="json")

        # skip links following when in sampled mode
        if self._mode == CrawlingModes.SAMPLED:
            return

        # following links
        for link in self._link_extractor.extract_links(response):
            if not link.url.startswith(f"{self.host}/{self._site_name}"):
                continue

            # skip static files
            url = urlparse(link.url)
            if re.match(r"^.*\.[a-zA-Z]{1,8}$", url.path):
                continue

            # skip ignored paths
            unquoted_path = unquote(url.path, encoding="utf-8", errors="replace")
            site_path = unquoted_path.split(self._site_name + "/")[1]
            if any([site_path.startswith(prefix) for prefix in IGNORED_PATH_PREFIX]):
                continue

            # skip processed links
            if link.url in self._processed_links:
                continue

            yield scrapy.Request(link.url, callback=self.parse)
