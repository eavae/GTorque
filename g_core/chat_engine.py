from llama_index.core import VectorStoreIndex
from llama_index.core.chat_engine.types import ChatMode
from g_core.llama_index_reranker import LlamaIndexReranker
from g_core.setup_globals import get_vector_store, get_memory
from g_core.prompts.default import (
    DEFAULT_CONDENSE_PROMPT_TEMPLATE,
    DEFAULT_CHAT_QA_SYSTEM_TMPL,
)
def buildChatEngine(db, uid) :
    print("step1: getting vector store")
    memory = get_memory(uid=uid)
    vector_store = get_vector_store(db)

    print("step2: build chat engine")
    index = VectorStoreIndex.from_vector_store(vector_store)
    reranker = LlamaIndexReranker(top_k=4)
    similarity_top_k = 16
    sparse_top_k = 16
    chat_engine = index.as_chat_engine(
        similarity_top_k=similarity_top_k,
        sparse_top_k=sparse_top_k,
        hybrid_top_k=similarity_top_k + sparse_top_k,
        memory=memory,
        chat_mode=ChatMode.CONDENSE_PLUS_CONTEXT,
        node_postprocessors=[reranker],
        condense_prompt=DEFAULT_CONDENSE_PROMPT_TEMPLATE,
        context_prompt=DEFAULT_CHAT_QA_SYSTEM_TMPL,
    )
    return chat_engine