import json
import tempfile
import argparse
import asyncio
from lxml import html
from scrapy.crawler import CrawlerProcess

from g_crawler.spiders import BWikiSpider, CrawlingModes, CONFIG_ROOT
from g_crawler.etree_tools import clean_html
from g_crawler.auto_trimmer import build_auto_trimmer, RuleStates
from g_crawler.mediawiki.template import update_site_meta

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
            tree = html.fragment_fromstring(data["html"])
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


def sample(args):
    assert args.provider == "biligame", "Only support biligame by now."

    asyncio.run(
        update_site_meta(
            args.name,
            fields=["categories", "samples"],
            mode="replace",
        )
    )


def analyse(args):
    assert args.provider == "biligame", "Only support biligame by now."

    extract_trim_rules(BWikiSpider, args.name)


def test(args):
    assert args.provider == "biligame", "Only support biligame by now."

    process = CrawlerProcess(
        settings={
            "ITEM_PIPELINES": {
                "g_crawler.spiders.MarkdownWriterPipeline": 1,
            }
        }
    )

    process.crawl(BWikiSpider, name=args.name, mode=CrawlingModes.SAMPLED)
    process.start()


def crawl(args):
    assert args.provider == "biligame", "Only support biligame by now."

    process = CrawlerProcess(
        settings={
            "ITEM_PIPELINES": {
                "g_crawler.spiders.OssWriterPipeline": 1,
            },
            "LOG_LEVEL": "WARNING",
        }
    )

    process.crawl(BWikiSpider, name=args.name)
    process.start()


def main():
    main_parser = argparse.ArgumentParser(
        prog="G-Crawler",
        description="A Crawler Tool Set for Gamepedia.",
        add_help=False,
    )
    main_parser.add_argument(
        "--provider",
        "-p",
        type=str,
        choices=["biligame"],
        default="biligame",
        required=False,
        help="the provider of the gamepedia site, only support biligame by now.",
    )
    main_parser.add_argument(
        "--name",
        "-n",
        type=str,
        required=True,
        help="the name of the gamepedia site, like ys（原神）, eldenring（艾尔登法环）",
    )

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    # subcommand: sample
    parser_sample = subparsers.add_parser(
        "sample",
        help="sample a subset from the site (aka, generate a config file)",
        parents=[main_parser],
    )
    parser_sample.set_defaults(func=sample)

    # subcommand: analyse
    parser_trim = subparsers.add_parser(
        "analyse",
        help="analyse sampled pages to generate trim rules",
        parents=[main_parser],
    )
    parser_trim.set_defaults(func=analyse)

    # subcommand: test
    parser_test = subparsers.add_parser(
        "test",
        help="test the crawler with sampled data",
        parents=[main_parser],
    )
    parser_test.set_defaults(func=test)

    # subcommand: crawl
    parser_crawl = subparsers.add_parser(
        "crawl",
        help="crawl the site with the trim rules",
        parents=[main_parser],
    )
    parser_crawl.set_defaults(func=crawl)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
