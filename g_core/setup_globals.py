import os
from functools import lru_cache
from llama_index.core.llms.llm import LLM
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, AsyncQdrantClient
from functools import partial

from g_core.llama_index_embedding import LlamaIndexEmbedding, RedisCache
from g_core.config.redis import get_redis
from g_core.llama_index_openai_ext_model import OpenAIExt


EMBEDDING_MODEL = "BAAI/bge-m3"

client = QdrantClient()
aclient = AsyncQdrantClient()

llm = OpenAIExt(
    model=os.getenv("G_BOT_CHAT_MODEL"),
    max_retries=3,
    max_tokens=1536,
    temperature=0.7,
)


embed_model = LlamaIndexEmbedding(
    cache=RedisCache(
        client=get_redis(),
        prefix="sparse_embedding_cache",
    ),
    embed_batch_size=16,
    text_batch_size=16,
    text_model=EMBEDDING_MODEL.replace("/", "__"),
)


def get_sparse_embedding_batch(texts):
    return embed_model.get_sparse_embedding_batch(texts)


async def aget_sparse_embedding_batch(texts):
    return await embed_model.aget_sparse_embedding_batch(texts)


def get_vector_store(collection_name: str, enable_hybrid=True, *args, **kwargs):
    return QdrantVectorStore(
        collection_name,
        client=client,
        aclient=aclient,
        enable_hybrid=enable_hybrid,
        sparse_doc_fn=get_sparse_embedding_batch,
        sparse_query_fn=aget_sparse_embedding_batch,
    )


@lru_cache
def get_prompt_helper(_llm: LLM = None) -> PromptHelper:
    if _llm is None:
        _llm = llm

    return PromptHelper.from_llm_metadata(
        _llm.metadata,
        tokenizer=_llm._tokenizer.encode if hasattr(_llm, "_tokenizer") else None,
    )


Settings.llm = llm
Settings.embed_model = embed_model
