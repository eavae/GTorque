from tokenizers import Tokenizer
from functools import lru_cache

from .constants import MAX_LENGTH_CONFIG, TEXT_EMBEDDING_FOLDER


def get_max_length(model_name: str):
    return MAX_LENGTH_CONFIG[model_name]


@lru_cache(maxsize=None)
def get_tokenizer(model_name: str):
    tokenizer_path = (
        f"{TEXT_EMBEDDING_FOLDER}/{model_name.replace('__', '/')}/tokenizer.json"
    )
    tokenizer: Tokenizer = Tokenizer.from_file(tokenizer_path)
    max_length = get_max_length(model_name)
    tokenizer.enable_padding()
    tokenizer.enable_truncation(max_length)
    return tokenizer
