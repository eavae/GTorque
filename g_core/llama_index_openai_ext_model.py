import os
from llama_index.core.base.llms.types import LLMMetadata
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai.utils import ALL_AVAILABLE_MODELS

# mapping from model name to pretrain model name in huggingface
PRETRAIN_MAPPING = {
    "deepseek-chat": "deepseek-ai/DeepSeek-V2-Chat",
}

CONTEXT_LENGTH_MAPPING = {
    "deepseek-chat": 32768,
    "qwen-long": 16384,  # assume
}

FUNCTION_CALLINGS = set()


class OpenAIExt(OpenAI):

    @property
    def _tokenizer(self):
        if self.model in ALL_AVAILABLE_MODELS:
            return super()._tokenizer

        from tokenizers import Tokenizer

        return Tokenizer.from_file(f"config/tokenizers/{self.model}.json")

    @property
    def metadata(self) -> LLMMetadata:
        if self.model in ALL_AVAILABLE_MODELS:
            return super().metadata

        return LLMMetadata(
            context_window=CONTEXT_LENGTH_MAPPING.get(self.model, 4096),
            num_output=self.max_tokens or -1,
            is_chat_model=True,
            is_function_calling_model=self.model in FUNCTION_CALLINGS,
            model_name=self.model,
        )
