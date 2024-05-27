import unittest
from redis import Redis
from llama_index.core.memory.types import ChatMessage

from g_core.memory.max_conversation_memory_buffer import (
    RedisChatStore,
    MaxConversationMemoryBuffer,
)


class RedisChatStoreTests(unittest.TestCase):
    def setUp(self):
        self.redis_client = Redis.from_url("redis://localhost:6379/0")
        self.chat_store = RedisChatStore(redis_client=self.redis_client)

    def tearDown(self):
        self.redis_client.flushall()

    def test_set_messages(self):
        key = "test_key"
        messages = [
            ChatMessage(content="Hello", role="assistant"),
            ChatMessage(content="Hi", role="user"),
        ]

        self.chat_store.set_messages(key, messages)

        stored_messages = self.redis_client.lrange(key, 0, -1)
        self.assertEqual(len(stored_messages), 2)

    def test_get_messages(self):
        key = "test_key"
        messages = [
            ChatMessage(content="Hello", role="assistant"),
            ChatMessage(content="Hi", role="user"),
        ]
        self.chat_store.set_messages(key, messages)

        retrieved_messages = self.chat_store.get_messages(key)

        self.assertEqual(len(retrieved_messages), 2)
        self.assertEqual(retrieved_messages[0].content, "Hello")
        self.assertEqual(retrieved_messages[1].content, "Hi")

    def test_add_message(self):
        key = "test_key"
        message = ChatMessage(content="Hello", role="assistant")

        self.chat_store.add_message(key, message)

        stored_messages = self.chat_store.get_messages(key)
        self.assertEqual(len(stored_messages), 1)
        self.assertEqual(stored_messages[0].content, "Hello")

    def test_delete_messages(self):
        key = "test_key"
        messages = [
            ChatMessage(content="Hello", role="assistant"),
            ChatMessage(content="Hi", role="user"),
        ]
        self.chat_store.set_messages(key, messages)

        deleted_messages = self.chat_store.delete_messages(key)

        stored_messages = self.chat_store.get_messages(key)
        self.assertEqual(len(stored_messages), 0)
        self.assertEqual(len(deleted_messages), 2)
        self.assertEqual(deleted_messages[0].content, "Hello")
        self.assertEqual(deleted_messages[1].content, "Hi")

    def test_delete_message(self):
        key = "test_key"
        messages = [
            ChatMessage(content="Hello", role="assistant"),
            ChatMessage(content="Hi", role="user"),
        ]
        self.chat_store.set_messages(key, messages)

        deleted_message = self.chat_store.delete_message(key, 0)

        stored_messages = self.chat_store.get_messages(key)
        self.assertEqual(len(stored_messages), 1)
        self.assertEqual(stored_messages[0].content, "Hi")
        self.assertEqual(deleted_message.content, "Hello")

    def test_delete_last_message(self):
        key = "test_key"
        messages = [
            ChatMessage(content="Hello", role="assistant"),
            ChatMessage(content="Hi", role="user"),
        ]
        self.chat_store.set_messages(key, messages)

        deleted_message = self.chat_store.delete_last_message(key)

        stored_messages = self.chat_store.get_messages(key)
        self.assertEqual(len(stored_messages), 1)
        self.assertEqual(stored_messages[0].content, "Hello")
        self.assertEqual(deleted_message.content, "Hi")

    def test_add_message_with_index(self):
        key = "test_key"
        messages = [
            ChatMessage(content="Hello", role="assistant"),
            ChatMessage(content="Hi", role="user"),
        ]
        self.chat_store.set_messages(key, messages)

        self.chat_store.add_message(
            key, ChatMessage(content="Hey", role="assistant"), -1
        )

        stored_messages = self.chat_store.get_messages(key)
        self.assertEqual(len(stored_messages), 3)
        self.assertEqual(stored_messages[0].content, "Hello")
        self.assertEqual(stored_messages[1].content, "Hey")
        self.assertEqual(stored_messages[2].content, "Hi")


class MaxConversationMemoryBufferTests(unittest.TestCase):
    def test_get_with_token_limit(self):
        memory_buffer = MaxConversationMemoryBuffer(
            token_limit=20,
            tokenizer_fn=lambda x: x,
            max_conversations=3,
            chat_store=RedisChatStore(),
            chat_store_key="test_key",
        )
        messages = [
            ChatMessage(content="Hello", role="assistant"),
            ChatMessage(content="Hi", role="user"),
            ChatMessage(content="How are you?", role="assistant"),
            ChatMessage(content="I'm good, thanks!", role="user"),
        ]
        memory_buffer.set(messages)

        chat_history = memory_buffer.get(initial_token_count=0)

        self.assertEqual(len(chat_history), 1)
        self.assertEqual(chat_history[0].content, "I'm good, thanks!")

        memory_buffer.reset()

    def test_get_with_conversation_limit(self):
        memory_buffer = MaxConversationMemoryBuffer(
            token_limit=1536,
            tokenizer_fn=lambda x: x,
            max_conversations=1,
            chat_store=RedisChatStore(),
            chat_store_key="test_key",
        )
        messages = [
            ChatMessage(content="Hello", role="assistant"),
            ChatMessage(content="Hi", role="user"),
            ChatMessage(content="How are you?", role="assistant"),
            ChatMessage(content="I'm good, thanks!", role="user"),
        ]
        memory_buffer.set(messages)

        chat_history = memory_buffer.get(initial_token_count=0)

        self.assertEqual(len(chat_history), 2)
        self.assertEqual(chat_history[0].content, "How are you?")
        self.assertEqual(chat_history[1].content, "I'm good, thanks!")

        memory_buffer.reset()
