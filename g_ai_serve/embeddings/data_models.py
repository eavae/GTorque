from enum import Enum
from pydantic import BaseModel
from typing import Union, List, Optional, Tuple

from g_ai_serve.core import Usage


class EmbeddingEncodingFormat(str, Enum):
    FLOAT = "float"
    BASE64 = "base64"


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str], List[List[str]]]
    model: str = "BAAI__bge-large-zh-v1.5"
    sparse_top_n: int = 128
    encoding_format: EmbeddingEncodingFormat = EmbeddingEncodingFormat.FLOAT
    user: Optional[str] = None


RerankerPair = Tuple[str, str]


class RerankerRequest(BaseModel):
    input: Union[RerankerPair, List[RerankerPair]]
    model: str = "bge-reranker-v2-m3"
    normalized: bool = True
    user: Optional[str] = None


class Embedding(BaseModel):
    embedding: List[float] | str
    index: int
    object: Optional[str] = "embedding"


class SparseEmbedding(BaseModel):
    indices: List[int]
    values: List[float]


class EmbeddingResponse(BaseModel):
    model: str
    data: List[Embedding]
    sparse: Optional[List[SparseEmbedding]]
    usage: Usage
    object: Optional[str] = "list"


class RerankerResponse(BaseModel):
    model: str
    data: List[float]
    usage: Usage
    object: Optional[str] = "list"
