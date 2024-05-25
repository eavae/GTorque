from qdrant_client import QdrantClient, AsyncQdrantClient

client = QdrantClient()
aclient = AsyncQdrantClient()


def get_qdrant_client():
    return client


def get_async_qdrant_client():
    return aclient
