import json
import tempfile
from lxml import html
from scrapy.crawler import CrawlerProcess

from g_crawler.spiders import BWikiSpider, CrawlingModes, CONFIG_ROOT
from g_crawler.etree_tools import clean_html
from g_crawler.auto_trimmer import build_auto_trimmer, RuleStates

auto_trimmer = build_auto_trimmer()


def extract_trim_rules(spider_cls, name):
    # using temp file to store the data
    with tempfile.NamedTemporaryFile(mode="w+") as f:
        # 1. crawling sampled data
        process = CrawlerProcess(
            settings={
                "FEEDS": {
                    f.name: {
                        "format": "jsonl",
                        "ensure_ascii": False,
                        "encoding": "utf-8",
                    }
                },
                "FEED_URI_PARAMS": "g_crawler.tools.uri_params",
            }
        )
        process.crawl(spider_cls, name=name, mode=CrawlingModes.SAMPLED)
        process.start()

        # 2. extract trim rules
        f.seek(0)
        drafts = []
        for line in f:
            data = json.loads(line)
            tree = html.fragment_fromstring(data["content"])
            tree = clean_html(tree)
            drafts.append({"tree": tree, "title": data["title"]})
        state = auto_trimmer.invoke(
            {"drafts": drafts, "rules": []},
            config={"recursion_limit": len(drafts) * 4},
        )
        rules = [
            rule["rule"]
            for rule in state["rules"]
            if rule["state"] == RuleStates.ACCEPTED
        ]

        # 3. write to config file
        config_file_path = f"{CONFIG_ROOT}/{name}/meta.json"
        config_data = json.load(open(config_file_path, "r"))
        config_data["trim_rules"] = rules
        json.dump(
            config_data, open(config_file_path, "w"), ensure_ascii=False, indent=2
        )


if __name__ == "__main__":
    extract_trim_rules(BWikiSpider, "eldenring")
