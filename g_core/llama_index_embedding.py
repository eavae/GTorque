import os
import base64
import httpx
import hashlib
import json
from typing import List, Dict, TYPE_CHECKING
from enum import Enum
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from retry import retry
from llama_index.core.embeddings.multi_modal_base import MultiModalEmbedding, Embedding
from llama_index.core.schema import ImageType, QueryBundle
from llama_index.core.bridge.pydantic import PrivateAttr, Field


TEXT_MODELS = {
    "BAAI__bge-large-zh-v1.5",
    "BAAI__bge-m3",
}
IMAGE_MODELS = {"laion__CLIP-ViT-bigG-14-laion2B-39B-b160k"}


class ModelTypes(str, Enum):
    IMAGE = "image"
    TEXT = "text"


QUERY_TEMPLATE = {
    "BAAI__bge-large-zh-v1.5": "为这个句子生成表示以用于检索相关文章：{query}"
}


class AbstractCache:

    def __getitem__(self, key) -> List[List[float]]:
        """indices, values"""
        raise NotImplementedError

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __contains__(self, key):
        raise NotImplementedError

    def get(self, key, default=None):
        raise NotImplementedError


class InMemoryCache:
    def __init__(self, max_size: int = 1024 * 1024):
        from lru import LRU

        self.cache = LRU(max_size)

    def __getitem__(self, key) -> List[List[float]]:
        return self.cache[key]

    def __setitem__(self, key, value):
        self.cache[key] = value

    def __contains__(self, key):
        return key in self.cache

    def get(self, key, default=None):
        return self.cache.get(key, default)


class RedisCache:
    if TYPE_CHECKING:
        from redis import Redis

    def __init__(
        self,
        client: "Redis",
        prefix: str = "sparse_embedding_cache",
        ttl: int = 60 * 60 * 24,
    ):
        self.cache = client
        self.prefix = prefix
        self.ttl = ttl

    def __getitem__(self, key) -> List[List[float]]:
        return json.loads(self.cache.get(f"{self.prefix}:{key}"))

    def __setitem__(self, key, value):
        self.cache.set(
            f"{self.prefix}:{key}",
            json.dumps(value),
            ex=self.ttl,
        )

    def __contains__(self, key):
        return self.cache.exists(key)

    def get(self, key, default=None):
        value = self.cache.get(f"{self.prefix}:{key}")
        return json.loads(value) if value is not None else default


class LlamaIndexEmbedding(MultiModalEmbedding):
    uni_format: bool = True
    text_model: str = "BAAI__bge-large-zh-v1.5"
    image_model: str = "laion__CLIP-ViT-bigG-14-laion2B-39B-b160k"
    text_batch_size: int = 8
    image_batch_size: int = 4
    sparse_top_n: int = 128

    base_url: str = Field(
        default_factory=lambda: os.environ.get(
            "G_EMBEDDING_BASE_URL", "http://localhost:8088"
        ),
    )
    path: str = "/v1/embeddings"
    access_token: str = Field(
        default_factory=lambda: os.environ.get("G_EMBEDDING_API_KEY", "")
    )

    timeout: int = 60  # seconds

    _cache: AbstractCache = PrivateAttr(InMemoryCache)

    def __init__(self, cache: AbstractCache = None, **data):
        super().__init__(**data)

        if cache:
            self._cache = cache
        else:
            self._cache = InMemoryCache()

    @property
    def model_type(self):
        if self.text_model in TEXT_MODELS:
            return ModelTypes.TEXT
        elif self.text_model in IMAGE_MODELS:
            return ModelTypes.IMAGE
        else:
            raise ValueError(f"Unsupported model {self.text_model}")

    @property
    def endpoint(self):
        if self.base_url.endswith("/") and self.path.startswith("/"):
            return f"{self.base_url[0:-1]}{self.path}"
        elif not self.base_url.endswith("/") and not self.path.startswith("/"):
            return f"{self.base_url}/{self.path}"
        return f"{self.base_url}{self.path}"

    def get_query_bundle(self, query: str):
        template = QUERY_TEMPLATE.get(self.text_model, None)
        if template:
            return QueryBundle(
                query_str=query,
                custom_embedding_strs=[template.format(query=query)],
            )
        return QueryBundle(query_str=query)

    def _get_hash_key(self, text: str):
        return hashlib.md5(f"{self.text_model}:{text}".encode("utf-8")).digest().hex()

    def _get_cached_embeddings(self, batch_query: List[str]) -> Dict:
        sparse_embeddings = [None] * len(batch_query)

        # 1. hash key
        hash_keys = [self._get_hash_key(text) for text in batch_query]

        # 2. fill by cached values and get no cached indices
        not_hit_indices = []
        for i, hash_key in enumerate(hash_keys):
            cached = self._cache.get(hash_key, None)
            if cached:
                sparse_embeddings[i] = cached
            else:
                not_hit_indices.append(i)

        return sparse_embeddings, hash_keys, not_hit_indices

    async def aget_sparse_embedding_batch(self, batch_query: List[str]) -> Dict:
        sparse_embeddings, hash_keys, not_hit_indices = self._get_cached_embeddings(
            batch_query
        )

        # 3. retrieve not hit queries
        await self.aget_text_embedding_batch([batch_query[i] for i in not_hit_indices])

        # 4. fill missing sparse embeddings
        for i in not_hit_indices:
            sparse_embeddings[i] = self._cache.get(hash_keys[i])

        # 5. return batch
        return ([x[0] for x in sparse_embeddings], [x[1] for x in sparse_embeddings])

    def get_sparse_embedding_batch(self, batch_query: List[str]) -> Dict:
        sparse_embeddings, hash_keys, not_hit_indices = self._get_cached_embeddings(
            batch_query
        )

        # 3. retrieve not hit queries
        self.get_text_embedding_batch([batch_query[i] for i in not_hit_indices])

        # 4. fill missing sparse embeddings
        for i in not_hit_indices:
            sparse_embeddings[i] = self._cache.get(hash_keys[i])

        # 5. return batch
        return ([x[0] for x in sparse_embeddings], [x[1] for x in sparse_embeddings])

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embedding_texts([query])[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return await self._a_embedding_texts([query])[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embedding_texts([text])[0]

    async def _aget_text_embedding(self, text: str) -> Embedding:
        return await self._a_embedding_texts([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for i in range(0, len(texts), self.text_batch_size):
            embeddings.extend(
                self._embedding_texts(texts[i : i + self.text_batch_size])  # noqa
            )
        return embeddings

    async def _aget_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        embeddings = []
        for i in range(0, len(texts), self.text_batch_size):
            embeddings.extend(
                await self._a_embedding_texts(
                    texts[i : i + self.text_batch_size]  # noqa
                )
            )
        return embeddings

    def _get_image_embeddings(
        self,
        images: List[ImageType],
    ) -> List[List[float]]:
        embeddings = []
        for i in range(0, len(images), self.image_batch_size):
            embeddings.extend(
                self._embedding_images(
                    images[i : i + self.image_batch_size],  # noqa
                )
            )
        return embeddings

    async def _aget_image_embeddings(
        self,
        images: List[ImageType],
    ) -> List[Embedding]:
        embeddings = []
        for i in range(0, len(images), self.image_batch_size):
            embeddings.extend(
                await self._a_embedding_images(
                    images[i : i + self.image_batch_size],  # noqa
                )
            )
        return embeddings

    def _get_image_embedding(self, image: ImageType) -> List[float]:
        return self._embedding_images([image])[0]

    async def _aget_image_embedding(self, image: ImageType) -> List[float]:
        return await self._a_embedding_images([image])[0]

    def _embedding_texts(self, texts: List[str]) -> List[List[float]]:
        return self._embedding(self.text_model, texts)

    async def _a_embedding_texts(self, texts: List[str]) -> List[float]:
        return await self._a_embedding(self.text_model, texts)

    def _embedding_images(self, images: List[ImageType]) -> List[List[float]]:
        # 这里需要进行编码,bytes -> base64
        encoded_images = []
        for image in images:
            if isinstance(image, str):
                encoded_images.append(image)
            else:
                if self.uni_format:
                    try:
                        image = Image.open(image)
                    except UnidentifiedImageError:
                        encoded_images.append(None)

                    image.thumbnail((448, 448))
                    image_bytes = BytesIO()
                    image.save(image_bytes, format="JPEG")
                    encoded_images.append(
                        f"data:image/jpeg;base64,{base64.b64encode(image_bytes.getvalue()).decode('utf-8')}"
                    )
                else:
                    encoded_images.append(
                        f"data:image/jpeg;base64,{base64.b64encode(image).decode('utf-8')}"
                    )

        return self._embedding(self.image_model, encoded_images)

    async def _a_embedding_images(self, images: List[str]) -> List[float]:
        return await self._a_embedding(self.image_model, images)

    @retry(exceptions=(httpx.RequestError), tries=3, delay=0.2)
    def _embedding(self, model: str, inputs: List[str]) -> List[float]:
        with httpx.Client() as client:
            resp = client.post(
                self.endpoint,
                json={
                    "model": model,
                    "input": inputs,
                    "sparse_top_n": self.sparse_top_n,
                },
                headers={
                    "Authorization": f"Bearer {self.access_token}",
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            embeddings = data["data"]
            if "sparse" in data:
                for s, sparse in zip(inputs, data["sparse"]):
                    key = hashlib.md5(s.encode("utf-8")).digest().hex()
                    self._cache[key] = [sparse["indices"], sparse["values"]]
            return [e["embedding"] for e in embeddings]

    @retry(exceptions=(httpx.RequestError), tries=3, delay=0.2)
    async def _a_embedding(self, model: str, inputs: List[str]) -> List[float]:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                self.endpoint,
                json={
                    "model": model,
                    "input": inputs,
                    "sparse_top_n": self.sparse_top_n,
                },
                headers={
                    "Authorization": f"Bearer {self.access_token}",
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            embeddings = data["data"]
            if "sparse" in data:
                for s, sparse in zip(inputs, data["sparse"]):
                    key = hashlib.md5(s.encode("utf-8")).digest().hex()
                    self._cache[key] = [sparse["indices"], sparse["values"]]
            return [e["embedding"] for e in embeddings]
