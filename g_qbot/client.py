import botpy
from botpy.message import Message, DirectMessage
from llama_index.core import VectorStoreIndex
from llama_index.core.chat_engine.types import ChatMode


from g_core.llama_index_reranker import LlamaIndexReranker
from g_core.setup_globals import get_vector_store, get_memory
from g_core.prompts.default import (
    DEFAULT_CONDENSE_PROMPT_TEMPLATE,
    DEFAULT_CHAT_QA_SYSTEM_TMPL,
)


class BotClient(botpy.Client):
    async def on_at_message_create(self, message: Message):
        await message.reply(
            content=f"机器人{self.robot.name}收到你的@消息了: {message.content}"
        )

    async def on_direct_message_create(self, message: DirectMessage):
        """
        此处为处理该事件的代码
        """
        uid = message.author.id
        memory = get_memory(uid=uid)

        vector_store = get_vector_store("test")
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
        response = await chat_engine.achat(message.content)

        await message.reply(content=response.response)

    async def on_direct_message_delete(self, message: DirectMessage):
        """
        此处为处理该事件的代码
        TODO: 撤销后应撤销机器人的回复并删除历史消息
        """
        pass
