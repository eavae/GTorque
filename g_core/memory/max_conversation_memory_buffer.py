import json
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.bridge.pydantic import Field, root_validator
from llama_index.core.llms.llm import LLM
from llama_index.core.memory.types import DEFAULT_CHAT_STORE_KEY, BaseMemory
from llama_index.core.storage.chat_store import BaseChatStore, SimpleChatStore
from llama_index.core.utils import get_tokenizer
from redis import Redis
from datetime import datetime


DEFAULT_TOKEN_LIMIT_RATIO = 0.75
DEFAULT_TOKEN_LIMIT = 3000


def _pack_message(message: ChatMessage) -> Dict[str, Any]:
    return json.dumps(
        {
            "message": message.dict(),
            "timestamp": int(datetime.now().timestamp()),
        }
    )


def _unpack_message(s: str | bytes) -> Tuple[ChatMessage, datetime]:
    d = json.loads(s)
    return ChatMessage.parse_obj(d["message"]), datetime.fromtimestamp(d["timestamp"])


class RedisChatStore(BaseChatStore):
    """
    A chat store that based on Redis's List data structure.

    NOTE: The saved messages never expire. It is the responsibility of the user to
    delete the messages when they are no longer needed.
    """

    redis_client: Redis = Field(
        default_factory=lambda: Redis.from_url("redis://localhost:6379/0"),
        description="Redis client.",
    )
    ttl: Optional[int] = Field(default=None, description="Time to live in seconds.")

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "RedisChatStore"

    def set_messages(self, key: str, messages: List[ChatMessage]):
        # delete if exists
        self.delete_messages(key, return_messages=False)

        # add messages
        json_messages = [_pack_message(m) for m in messages]
        self.redis_client.rpush(key, *json_messages)

        if self.ttl:
            self.redis_client.expire(key, self.ttl)

    def get_messages(self, key: str, start=0, end=-1) -> List[ChatMessage]:
        if end == -1:
            warnings.warn("Retrieving all messages from Redis can be slow. ")

        messages = self.redis_client.lrange(key, start, end)
        return [_unpack_message(m)[0] for m in messages]

    def add_message(self, key: str, message: ChatMessage, idx: Optional[int] = None):
        if idx is None:
            self.redis_client.rpush(key, _pack_message(message))
        else:
            self._insert_element_at_index(key, idx, message)

        if self.ttl:
            self.redis_client.expire(key, self.ttl)

    def _insert_element_at_index(self, key: str, idx: int, message: ChatMessage):
        # get the current message at the index
        current_message = self.redis_client.lindex(key, idx)
        if current_message is None:
            warnings.warn(
                f"Index {idx} does not exist in the list. Pushing to the end. "
            )
            return self.redis_client.rpush(key, _pack_message(message))

        new_message = _pack_message(message)

        # push the current message back to the list
        self.redis_client.linsert(key, "BEFORE", current_message, new_message)

    def delete_messages(
        self, key: str, return_messages=True
    ) -> List[ChatMessage] | None:
        messages = None
        if return_messages:
            messages = self.get_messages(key)
        self.redis_client.delete(key)
        return messages

    def delete_message(self, key: str, idx: int) -> ChatMessage | None:
        message = self.redis_client.lindex(key, idx)
        if message is not None:
            self.redis_client.lrem(key, 1, message)
        return _unpack_message(message)[0]

    def delete_last_message(self, key: str) -> ChatMessage | None:
        message = self.redis_client.rpop(key)
        return _unpack_message(message)[0]

    def get_keys(self) -> List[str]:
        return self.redis_client.keys()


class MaxConversationMemoryBuffer(BaseMemory):
    """Simple buffer for storing chat history."""

    token_limit: int = 1536
    tokenizer_fn: Callable[[str], List] = Field(
        default_factory=get_tokenizer,
        exclude=True,
    )
    max_conversations: int = 3
    chat_store: RedisChatStore = Field(default_factory=RedisChatStore)
    chat_store_key: str = Field(default=DEFAULT_CHAT_STORE_KEY)

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "ChatMemoryBuffer"

    @root_validator(pre=True)
    def validate_memory(cls, values: dict) -> dict:
        # Validate token limit
        token_limit = values.get("token_limit", -1)
        if token_limit < 1:
            raise ValueError("Token limit must be set and greater than 0.")

        # Validate tokenizer -- this avoids errors when loading from json/dict
        tokenizer_fn = values.get("tokenizer_fn", None)
        if tokenizer_fn is None:
            values["tokenizer_fn"] = get_tokenizer()

        return values

    @classmethod
    def from_defaults(
        cls,
        chat_history: Optional[List[ChatMessage]] = None,
        llm: Optional[LLM] = None,
        chat_store: Optional[BaseChatStore] = None,
        chat_store_key: str = DEFAULT_CHAT_STORE_KEY,
        token_limit: Optional[int] = None,
        tokenizer_fn: Optional[Callable[[str], List]] = None,
    ) -> "MaxConversationMemoryBuffer":
        """Create a chat memory buffer from an LLM."""
        if llm is not None:
            context_window = llm.metadata.context_window
            token_limit = token_limit or int(context_window * DEFAULT_TOKEN_LIMIT_RATIO)
        elif token_limit is None:
            token_limit = DEFAULT_TOKEN_LIMIT

        if chat_history is not None:
            chat_store = chat_store or RedisChatStore()
            chat_store.set_messages(chat_store_key, chat_history)

        return cls(
            token_limit=token_limit,
            tokenizer_fn=tokenizer_fn or get_tokenizer(),
            chat_store=chat_store or RedisChatStore(),
            chat_store_key=chat_store_key,
        )

    def to_string(self) -> str:
        """Convert memory to string."""
        return self.json()

    @classmethod
    def from_string(cls, json_str: str) -> "MaxConversationMemoryBuffer":
        """Create a chat memory buffer from a string."""
        dict_obj = json.loads(json_str)
        return cls.from_dict(dict_obj)

    def to_dict(self, **kwargs: Any) -> dict:
        """Convert memory to dict."""
        return self.dict()

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], **kwargs: Any
    ) -> "MaxConversationMemoryBuffer":
        from llama_index.core.storage.chat_store.loading import load_chat_store

        # NOTE: this handles backwards compatibility with the old chat history
        if "chat_history" in data:
            chat_history = data.pop("chat_history")
            chat_store = RedisChatStore()
            chat_store.set_messages(DEFAULT_CHAT_STORE_KEY, chat_history)
            data["chat_store"] = chat_store

        elif "chat_store" in data:
            chat_store = data.pop("chat_store")
            chat_store = load_chat_store(chat_store)
            data["chat_store"] = chat_store

        return cls(**data)

    def get(self, initial_token_count: int = 0, **kwargs: Any) -> List[ChatMessage]:
        """Get chat history."""
        if initial_token_count > self.token_limit:
            raise ValueError("Initial token count exceeds token limit")

        batch_size = self.max_conversations * 2
        max_user_bot_message_count = self.max_conversations * 2

        chat_history = []
        batch_idx = 0
        user_bot_message_count = 0
        token_count = 0
        tmp_message_queue = []
        while True:
            if len(tmp_message_queue) == 0:
                batch_of_messages = self.chat_store.get_messages(
                    self.chat_store_key,
                    start=-(batch_idx + 1) * batch_size - 1,
                    end=-batch_idx * batch_size - 1,
                )
                batch_idx += 1

                for m in batch_of_messages:
                    tmp_message_queue.append(m)

            if len(tmp_message_queue) == 0:
                break

            message = tmp_message_queue.pop()
            message_token = len(self.tokenizer_fn(message.content))

            if token_count + message_token > self.token_limit:
                break

            # add to chat history
            chat_history.append(message)

            # update
            token_count += message_token
            if message.role == MessageRole.USER:
                user_bot_message_count += 1
            elif message.role == MessageRole.ASSISTANT:
                user_bot_message_count += 1

            if user_bot_message_count >= max_user_bot_message_count:
                break

        return list(reversed(chat_history))

    def get_all(self) -> List[ChatMessage]:
        """Get all chat history."""
        return self.chat_store.get_messages(self.chat_store_key)

    def put(self, message: ChatMessage) -> None:
        """Put chat history."""
        # ensure everything is serialized
        self.chat_store.add_message(self.chat_store_key, message)

    def set(self, messages: List[ChatMessage]) -> None:
        """Set chat history."""
        self.chat_store.set_messages(self.chat_store_key, messages)

    def reset(self) -> None:
        """Reset chat history."""
        self.chat_store.delete_messages(self.chat_store_key)

    def _token_count_for_messages(self, messages: List[ChatMessage]) -> int:
        if len(messages) <= 0:
            return 0

        msg_str = " ".join(str(m.content) for m in messages)
        return len(self.tokenizer_fn(msg_str))
