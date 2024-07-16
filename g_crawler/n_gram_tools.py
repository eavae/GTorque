import re
from typing import List
from jieba import cut

MAX_N_GRAM = 10
MIN_N_GRAM = 3


def _deduplicate_text_array_with_n_gram(texts: List[str], n):
    results = texts[:n]

    i = n
    while i < len(texts):
        gram = "".join(texts[i : i + n])  # noqa
        prev_gram = "".join(results[-n:])

        # 跳过数字和markdown中的符号
        if any(char.isdigit() or char in "[](){}|\n" for char in gram):
            results.append(texts[i])
            i += 1
            continue

        # 跳过相同的n-gram
        if gram == prev_gram:
            i += n
            continue

        # 跳过任意多空格
        if results[-1] == " " and texts[i] == " ":
            i += 1
            continue

        results.append(texts[i])
        i += 1

    return results


def deduplicate_text(
    text: str,
    min_n_gram: int = MIN_N_GRAM,
    max_n_gram: int = MAX_N_GRAM,
    trim_space: bool = True,
):
    """using n-gram to deduplicate text

    Args:
        text (str): _description_
        min_n_gram (int, optional): _description_. Defaults to MIN_N_GRAM.
        max_n_gram (int, optional): _description_. Defaults to MAX_N_GRAM.
        trim_space (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    tokens = list(cut(text))
    for n in range(max_n_gram, min_n_gram - 1, -1):
        tokens = _deduplicate_text_array_with_n_gram(tokens, n)
    text = "".join(tokens)

    # 将多个连续空格替换为一个空格
    if trim_space and min_n_gram >= 2:
        text = re.sub(r" +", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

    return text
