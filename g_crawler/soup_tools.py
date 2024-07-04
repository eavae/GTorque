import bs4
import re
from bs4 import BeautifulSoup
from copy import deepcopy, copy
from tokenizers import Tokenizer
from urllib.parse import unquote

from g_crawler.tools import find_most_repeated_sub_sequence_html
from g_crawler.html_constants import INLINE_ELEMENTS, INTERACTIVE_ELEMENTS


def _decode_url(element):
    if hasattr(element, "attrs"):
        if "href" in element.attrs:
            element.attrs["href"] = unquote(element.attrs["href"])
        if "src" in element.attrs:
            element.attrs["src"] = unquote(element.attrs["src"])


def decode_url(soup: BeautifulSoup):
    deep_first_travel(soup, _decode_url)
    return soup


def _clean_html_with_soup(element):
    # 移除注释等
    if isinstance(element, bs4.element.PreformattedString):
        element.extract()
        return

    # 移除交互元素
    if hasattr(element, "name") and element.name in INTERACTIVE_ELEMENTS:
        element.extract()
        return

    # 移除空白元素
    if (
        hasattr(element, "name")
        and element.name != "img"
        and element.get_text().strip() == ""
    ):
        element.extract()
        return

    # 移除多余属性
    if hasattr(element, "attrs"):
        element.attrs = {
            key: value
            for key, value in element.attrs.items()
            if key in ["class", "id", "title", "alt", "href", "src"]
        }


def clean_html(soup: BeautifulSoup):
    deep_first_travel(soup, _clean_html_with_soup)
    return soup


def extract_html_structure(soup: BeautifulSoup):
    soup = deepcopy(soup)
    soup = clean_html(soup)

    for element in soup.find_all():
        if isinstance(element, bs4.element.Tag):
            # only keep class and id attributes
            if element.attrs:
                element.attrs = {
                    key: value for key, value in element.attrs.items() if key == "class"
                }

            if element.name in INLINE_ELEMENTS:
                element.extract()
                continue

    # remove text nodes
    for element in soup.find_all(text=True):
        element.extract()

    return soup


def deep_first_travel(element: bs4.element.Tag, callback):
    if hasattr(element, "contents"):
        for child in list(
            element.children
        ):  # !important, list() is necessary to avoid skipping children
            deep_first_travel(child, callback)

    callback(element)


def keep_unique_structure(element: bs4.element.Tag):
    if not isinstance(element, bs4.element.Tag):
        return

    children = list(element.children)
    if not children or len(children) == 1:
        return

    # keep the first 2 children if they are table and tr
    is_table_with_tr = element.name == "table" and children[0].name == "tr"
    is_tbody_with_tr = element.name == "tbody" and children[0].name == "tr"
    if is_table_with_tr or is_tbody_with_tr:
        element.clear()
        element.extend(children[:2])
        return

    # keep the first child if they are ul and li
    is_ul_with_li = element.name == "ul" and children[0].name == "li"
    is_ol_with_li = element.name == "ol" and children[0].name == "li"
    if is_ul_with_li or is_ol_with_li:
        element.clear()
        element.append(children[0])
        return

    # if every child has the same structure, keep the first one
    str_children = [str(child) for child in children]
    for i in range(len(str_children) - 1):
        if not re.match(r"^<\w+\s+class=", str_children[i]):
            continue

        current = str_children[i]
        # compare to the rest of the children
        if all(current == child for child in str_children[i + 1 :]):  # noqa
            element.clear()
            element.extend(children[: i + 1])
            return

    # test if multiple children have the same structure
    repeats = find_most_repeated_sub_sequence_html(str_children)
    if repeats:
        remove_indices = set()
        for start, end in repeats[1:]:
            remove_indices.update(range(start, end))

        keep_indices = set(range(len(children))) - remove_indices
        keep_children = [children[i] for i in keep_indices]

        element.clear()
        element.extend(keep_children)


def _is_same_element(e1: bs4.element.Tag, e2: bs4.element.Tag):
    if not isinstance(e1, bs4.element.Tag):
        return False

    if not isinstance(e2, bs4.element.Tag):
        return False

    if e1.name != e2.name:
        return False

    if e1.attrs or e2.attrs:
        e1_attrs = e1.attrs or {}
        e2_attrs = e2.attrs or {}

        if e1_attrs.get("class") != e2_attrs.get("class"):
            return False

    return True


def prune_by_structure(origin: BeautifulSoup, structure: BeautifulSoup):
    """根据结构修剪原始的 html 树。"""
    assert _is_same_element(
        origin, structure
    ), "The structure is not the same as the origin."

    # 检查是否是叶子节点
    if not origin.contents or not structure.contents:
        return

    # 递归地修剪
    origin_i = 0
    structure_i = 0

    while origin_i < len(origin.contents) and structure_i < len(structure.contents):
        origin_child = origin.contents[origin_i]
        structure_child = structure.contents[structure_i]

        if _is_same_element(origin_child, structure_child):
            prune_by_structure(origin_child, structure_child)

            origin_i += 1
            structure_i += 1
            continue

        if origin_child:
            origin_child.extract()
            continue

    # 删除多余的节点
    origin.contents = origin.contents[:origin_i]


def prune_by_tokens(
    tokenizer: Tokenizer,
    soup: BeautifulSoup,
    max_tokens: int,
    reversed: bool = False,
):
    total_token = len(tokenizer.encode(str(soup)).ids)

    # 如果总长度小于 max_tokens，不需要修剪
    if total_token <= max_tokens:
        return

    # 如果是文字子节点，保留
    if isinstance(soup, bs4.element.NavigableString):
        return

    # 如果没有子节点，删除
    if not soup.contents:
        soup.extract()
        return

    self_node = copy(soup)
    self_node.clear()
    self_tokens = len(tokenizer.encode(str(self_node)).ids)
    required_tokens = max_tokens - self_tokens

    children = list(soup.children)
    if reversed:
        children = reversed(children)

    acc_tokens = 0
    for idx, child in enumerate(children):
        child_tokens = len(tokenizer.encode(str(child)).ids)
        if acc_tokens + child_tokens > required_tokens:
            break
        acc_tokens += child_tokens

    # 删除多余的节点
    if reversed:
        soup.contents = soup.contents[idx:]
    else:
        soup.contents = soup.contents[: idx + 1]

    # 递归修剪
    prune_by_tokens(tokenizer, child, required_tokens - acc_tokens, reversed=reversed)

    return soup


def extract_left_subset(
    soup: BeautifulSoup,
    tokenizer: Tokenizer,
    max_tokens: int = 2048,
):
    soup = deepcopy(soup)
    prune_by_tokens(tokenizer, soup, max_tokens, reversed=False)
    return soup
