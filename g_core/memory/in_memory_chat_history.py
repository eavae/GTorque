from typing import List
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage


class InMemoryChatHistory(InMemoryChatMessageHistory):
    k = 5

    async def aget_messages(self) -> List[BaseMessage]:
        return self.messages[-self.k * 2 :] if self.k > 0 else []  # noqa
