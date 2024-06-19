import aiohttp
import pandas as pd
import os
import json
import re
import random
from functools import lru_cache
from tqdm import tqdm
from retry import retry


@retry(tries=3, delay=1)
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


@retry(tries=3, delay=1)
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
        "prop": "revisions",
    }
    rvprop = "content" if with_content else None
    if rvprop:
        params["rvprop"] = rvprop

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            data = await response.json()

    pages = []
    for page in data["query"]["pages"].values():
        revisions = page["revisions"]
        content = revisions[0]["*"] if with_content and len(revisions) > 0 else None
        page = dict(pageid=page["pageid"], title=page["title"])
        if with_content:
            page["content"] = content
        pages.append(page)
    return pages


@lru_cache
def get_exclude_categories_zero_shot_chain():
    from langchain.chat_models.openai import ChatOpenAI
    from langchain.output_parsers.list import CommaSeparatedListOutputParser
    from langchain.prompts import PromptTemplate

    llm = ChatOpenAI(model="deepseek-coder", temperature=0.1, max_tokens=1024)
    output_parser = CommaSeparatedListOutputParser()
    prompt_template = PromptTemplate.from_template(
        (
            "你是一个游戏玩家，你需要从`给定类别`中剔除与游戏无关的分类，以免浪费时间。"
            "比如，诸如：左侧目录, 待完善, 待补充, 施工中, 未实装, 沙盒, 配音演员等类别名称与游戏无关，还有意义不明的数字、英文单词等。你可以根据你的经验进行扩展。\n"
            "\n"
            "给定类别：\n"
            "{categories}\n"
            "\n"
            "输出格式要求：\n"
            "{format_instructions}\n"
            "\n"
            "需被剔除的类别：\n"
        ),
        partial_variables={
            "format_instructions": output_parser.get_format_instructions()
        },
    )
    return prompt_template | llm | output_parser


def get_likert_scale_chain():
    from langchain.chains.llm import LLMChain
    from langchain.chat_models.openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
    from langchain.schema.messages import SystemMessage
    from langchain_core.prompts.chat import MessagesPlaceholder
    from langchain.memory.buffer_window import ConversationBufferWindowMemory

    memory = ConversationBufferWindowMemory(k=3, return_messages=True)
    llm = ChatOpenAI(model="deepseek-coder", temperature=0.01, max_tokens=1024)
    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    "你是一名游戏玩家，你需要对提供的材料进行评分(1-5之间的小数)。评分规则：5. 非常有价值，4. 有一定价值，3. 不一定有价值，2. 好像没啥价值，1. 毫无价值。某些材料具有时效性，请降低评分。评分时，你只需回复数字，无需任何解释。给定的材料节选自游戏相关的内容，无需关心材料的完整性和格式问题。"
                )
            ),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template(
                "请对以下内容进行评分（仅给出数字，不要解释）：\n{content}"
            ),
        ]
    )
    return LLMChain(memory=memory, llm=llm, prompt=prompt_template)


async def filter_categories_by_llm(categories: list[dict[str, int]]):
    """exclude categories."""

    # exclude with zero shot
    chain = get_exclude_categories_zero_shot_chain()
    to_be_excluded = set(
        await chain.ainvoke(
            {"categories": ", ".join([x["cat_name"] for x in categories])}
        )
    )
    categories = [
        category
        for category in categories
        if category["cat_name"] not in to_be_excluded
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
        content = page["content"][:1024]
        response = await chain.ainvoke({"content": content})
        # using regex, ^\d+(\.\d+)?
        score = re.match(r"^.*(\d+\.?\d*)", response["text"]).groups()[0]
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


async def update_site_meta(site: str, field: str = "categories"):
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
        categories = await filter_categories_by_llm(categories)
        meta_data[field] = categories

    with open(site_meta_file_path, "w") as f:
        f.write(json.dumps(meta_data, indent=2, ensure_ascii=False))


async def main():
    await update_site_meta("ys")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
