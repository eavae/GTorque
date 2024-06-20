import aiohttp
from aiohttp.client_exceptions import (
    ServerConnectionError,
    ServerTimeoutError,
    ServerDisconnectedError,
)
import pandas as pd
import os
import json
import re
import random
from functools import lru_cache
from tqdm import tqdm
from retry_async import retry
from datetime import datetime

retry_exceptions = (ServerDisconnectedError, ServerConnectionError, ServerTimeoutError)


@retry(
    exceptions=retry_exceptions,
    tries=3,
    delay=1,
    backoff=5,
    max_delay=10,
    is_async=True,
)
async def get_all_categories(
    site: str,
    limit: int = 500,
):
    """Get all categories from a MediaWiki site."""
    url = f"https://wiki.biligame.com/{site}/api.php"
    params = {
        "action": "query",
        "list": "allcategories",
        "format": "json",
        "aclimit": limit,
        "acprop": "size",
        "acmin": 1,
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            data = await response.json()

    return [
        dict(cat_name=category["*"], n_pages=category["pages"])
        for category in data["query"]["allcategories"]
        if category["pages"] > 0
    ]


@retry(
    exceptions=retry_exceptions,
    tries=3,
    delay=1,
    backoff=5,
    max_delay=10,
    is_async=True,
)
async def get_pages_by_category(
    site: str,
    category: str,
    limit: int = 10,
    with_content: bool = False,
):
    url = f"https://wiki.biligame.com/{site}/api.php"
    params = {
        "action": "query",
        "generator": "categorymembers",
        "format": "json",
        "gcmlimit": limit,
        "gcmtitle": f"分类:{category}",
        "prop": "revisions|info",
        "gcmsort": "timestamp",
        "gcmdir": "desc",
        "rvslots": "main",
    }

    rvprop = ["timestamp"]
    if with_content:
        rvprop.append("content")
    rvprop = "|".join(rvprop)

    params["rvprop"] = rvprop
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            data = await response.json()

    pages = []
    for page in data["query"]["pages"].values():
        if not page.get("revisions"):
            continue

        revision = page["revisions"][0]
        if with_content:
            page["content"] = revision["slots"]["main"]["*"]

        # !important 使用 revision["timestamp"] 而不是 page["touched"]
        page["timestamp"] = revision["timestamp"]
        pages.append(page)
    return pages


@lru_cache
def get_game_related_categories_zero_shot_chain():
    from langchain.chat_models.openai import ChatOpenAI
    from langchain.output_parsers.list import CommaSeparatedListOutputParser
    from langchain.prompts import PromptTemplate

    llm = ChatOpenAI(model="deepseek-chat", temperature=0.01, max_tokens=1024)
    output_parser = CommaSeparatedListOutputParser()
    prompt_template = PromptTemplate.from_template(
        (
            "从`给定类别`中选择与游戏内容、玩法、世界观、角色、技能、武器等相关要素有关的类别。按给定顺序逐个检查，并依次输出。"
            "比如，左侧目录、待完善、待补充、施工中、未实装、沙盒、配音演员等类别与游戏无关，意义不明的数字、英文单词等与游戏无关。结合你的经验进行谨慎的判断，如果你不是很确定，则保留该类别。\n"
            "\n"
            "给定类别：\n"
            "{categories}\n"
            "\n"
            "输出格式要求：\n"
            "{format_instructions}\n"
            "\n"
            "与游戏相关的类别：\n"
        ),
        partial_variables={
            "format_instructions": output_parser.get_format_instructions()
        },
    )
    return prompt_template | llm | output_parser


def get_likert_scale_chain():
    from langchain.chat_models.openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
    from langchain.schema.messages import SystemMessage
    from langchain_core.prompts.chat import MessagesPlaceholder
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from g_core.memory.in_memory_chat_history import InMemoryChatHistory

    memory = InMemoryChatHistory(k=0)
    llm = ChatOpenAI(model="deepseek-chat", temperature=0.01, max_tokens=1024)
    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    "你是一名硬核的游戏玩家，你需要对提供的材料进行打分(1-5之间的小数)，从而帮助你从该游戏中获得更多乐趣。评分规则：5. 非常感兴趣，4. 有点感兴趣，3. 说不上来，2. 不太感兴趣，1. 毫无兴趣。某些材料具有时效性、或转载自其它网站、或主要内容是视频或图片、或与游戏无关，请降低评分。评分时，你只需回复数字，无需任何解释。给定的材料节选自游戏相关的内容，你不用将重点放在材料的完整性和格式问题。当前时间是"
                    + datetime.now().strftime("%Y-%m-%d")
                )
            ),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template(
                "请对以下内容进行评分（仅给出数字，不要解释）：\n{content}"
            ),
        ]
    )
    return RunnableWithMessageHistory(
        prompt_template | llm,
        lambda x: memory,
        input_messages_key="content",
        history_messages_key="history",
    )


async def filter_categories_by_llm(categories: list[dict[str, int]]):
    """exclude categories."""

    # exclude with zero shot
    chain = get_game_related_categories_zero_shot_chain()
    to_be_includes = set(
        await chain.ainvoke(
            {"categories": ", ".join([x["cat_name"] for x in categories])}
        )
    )
    categories = [
        category for category in categories if category["cat_name"] in to_be_includes
    ]

    # exclude with 3 contents in this cat.
    pairs = []
    for category in categories:
        pages = await get_pages_by_category(
            "ys", category["cat_name"], limit=10, with_content=True
        )
        pages = random.sample(pages, min(3, len(pages)))
        for page in pages:
            pairs.append((category["cat_name"], page))
    random.shuffle(pairs)

    # get likert scale
    chain = get_likert_scale_chain()
    triples = []
    for cat_name, page in tqdm(pairs, desc="Likert Scoring"):
        content = page["content"][:512]
        created_at = page["touched"]
        title = page["title"]

        response = await chain.ainvoke(
            {"content": f"标题：{title}\n\n创建时间：{created_at}\n\n{content}"},
            config={"configurable": {"session_id": "ys-likert-scoring"}},
        )
        # using regex, ^\d+(\.\d+)?
        score = re.match(r"^.*(\d+\.?\d*)", response.content).groups()[0]
        triples.append((cat_name, page, float(score)))

    # group by category and avg score using pandas
    df = pd.DataFrame(triples, columns=["category", "page", "score"])
    df = df.groupby("category").agg(score=("score", "mean"))

    # filter by score
    df = df[df["score"] >= 3.5]

    # join with original categories
    categories = pd.DataFrame(categories)
    categories = categories.merge(df, left_on="cat_name", right_index=True)
    return categories.to_dict(orient="records")


async def update_site_meta(
    site: str,
    field: str = "categories",
    mode: str = "upsert",
):
    site_meta_file_path = f"configs/{site}/meta.json"

    # if folder not exists, create it
    os.makedirs(os.path.dirname(site_meta_file_path), exist_ok=True)
    # if file not exist, create it
    if not os.path.exists(site_meta_file_path):
        with open(site_meta_file_path, "w") as f:
            f.write(json.dumps({}, indent=2, ensure_ascii=False))

    meta_data = json.loads(open(site_meta_file_path).read())
    if field == "categories":
        categories = await get_all_categories("ys")

        # 该类别下至少有 2 篇文章
        categories = [x for x in categories if x["n_pages"] >= 2]
        categories = await filter_categories_by_llm(categories)

        if mode == "upsert":
            old_categories = meta_data.get(field, [])
            old_categories = {x["cat_name"]: x for x in old_categories}
            new_categories = {x["cat_name"]: x for x in categories}
            old_categories.update(new_categories)
            categories = list(old_categories.values())

        # sort by category name
        categories = sorted(categories, key=lambda x: x["cat_name"])
        meta_data[field] = categories

    with open(site_meta_file_path, "w") as f:
        f.write(json.dumps(meta_data, indent=2, ensure_ascii=False))


async def main():
    await update_site_meta("ys", mode="replace")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
