from lxml import etree
from lxml.html import HtmlElement, HtmlComment
from tokenizers import Tokenizer
from copy import deepcopy
from urllib.parse import unquote
from typing import List

from g_crawler.html_constants import INTERACTIVE_ELEMENTS


def post_order_traversal(tree: HtmlElement, func):
    for ele in tree.iterchildren():
        post_order_traversal(ele, func)

    func(tree)


def _clean_html(ele: HtmlElement):
    # 移除非元素的节点
    if not isinstance(ele, HtmlElement):
        ele.getparent().remove(ele)
        return

    if isinstance(ele, HtmlComment):
        ele.getparent().remove(ele)
        return

    # 移除交互元素
    if ele.tag in INTERACTIVE_ELEMENTS:
        ele.getparent().remove(ele)
        return

    # 移除空白元素
    if hasattr(ele, "tag") and ele.tag != "img" and not ele.text_content().strip():
        ele.getparent().remove(ele)
        return

    # 移除多余属性
    if ele.attrib:
        for key in list(ele.attrib.keys()):
            if key not in ["class", "id", "title", "alt", "href", "src"]:
                del ele.attrib[key]


def clean_html(ele: HtmlElement):
    post_order_traversal(ele, _clean_html)
    return ele


def remove_children(ele: HtmlElement):
    for child in ele.getchildren():
        ele.remove(child)
    return ele


def to_string(ele: HtmlElement, pretty_print=False):
    html = etree.tostring(ele, encoding="utf-8").decode("utf-8")
    if pretty_print:
        from bs4 import BeautifulSoup

        return BeautifulSoup(html, "html.parser").prettify()
    return html


def prune_by_tokens(
    tokenizer: Tokenizer,
    ele: HtmlElement,
    max_tokens: int,
    reversed: bool = False,
):
    if ele is None:
        return

    # 如果总长度小于 max_tokens，不需要修剪
    total_token = len(tokenizer.encode(to_string(ele)).ids)
    if total_token <= max_tokens:
        return

    # check children
    children = ele.getchildren()
    remove_children(ele)
    self_tokens = len(tokenizer.encode(to_string(ele)).ids)
    required_tokens = max_tokens - self_tokens
    if reversed:
        children = reversed(children)

    # no children
    if len(children) == 0:
        return

    acc_tokens = 0
    for idx, child in enumerate(children):
        child_tokens = len(tokenizer.encode(to_string(child)).ids)
        if acc_tokens + child_tokens > required_tokens:
            break
        acc_tokens += child_tokens

    # 保留需要的子节点
    if reversed:
        ele.extend(reversed(children[: idx + 1]))
    else:
        ele.extend(children[: idx + 1])

    # 递归修剪
    prune_by_tokens(tokenizer, child, required_tokens - acc_tokens, reversed=reversed)

    return ele


def extract_left_subtree(
    tokenizer: Tokenizer,
    element: HtmlElement,
    max_tokens: int = 2048,
):
    element = deepcopy(element)
    prune_by_tokens(tokenizer, element, max_tokens, reversed=False)
    return element


def _decode_url(element: HtmlElement):
    if element.attrib:
        if "href" in element.attrib:
            element.attrib["href"] = unquote(element.attrib["href"])

        if "src" in element.attrib:
            element.attrib["src"] = unquote(element.attrib["src"])


def decode_url(element: HtmlElement):
    post_order_traversal(element, _decode_url)
    return element


def apply_trim_rules(root: HtmlElement, rules: List[str]):
    for rule in rules:
        for ele in root.xpath(rule):
            ele.getparent().remove(ele)
    return root
