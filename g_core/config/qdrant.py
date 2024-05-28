import os
from qdrant_client import QdrantClient, AsyncQdrantClient

client = QdrantClient(
    os.getenv("QDRANT_URL", "localhost"),
)
aclient = AsyncQdrantClient(
    os.getenv("QDRANT_URL", "localhost"),
)


def get_qdrant_client():
    return client


def get_async_qdrant_client():
    return aclient
