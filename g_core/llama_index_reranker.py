import os
import httpx
from typing import List, Optional, Tuple
from retry import retry

from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle


SUPPORTED_MODELS = [
    "BAAI__bge-reranker-v2-m3",
]


class LlamaIndexReranker(BaseNodePostprocessor):
    """An Restful API based reranker."""

    model: str = Field(
        description="Sentence transformer model name.",
        default="BAAI__bge-reranker-v2-m3",
    )
    top_k: int = Field(
        description="Number of nodes to return sorted by score.", default=3
    )

    base_url: str = Field(
        default_factory=lambda: os.environ.get(
            "G_EMBEDDING_BASE_URL", "http://localhost:8088"
        ),
    )
    path: str = "/v1/rerankings"
    access_token: str = Field(
        default_factory=lambda: os.environ.get("G_EMBEDDING_API_KEY", ""),
    )
    timeout: int = 60  # seconds

    keep_retrieval_score: bool = Field(
        default=False,
        description="Whether to keep the retrieval score in metadata.",
    )

    @classmethod
    def class_name(cls) -> str:
        return "LlamaIndexReranker"

    @property
    def endpoint(self):
        if self.base_url.endswith("/") and self.path.startswith("/"):
            return f"{self.base_url[0:-1]}{self.path}"
        elif not self.base_url.endswith("/") and not self.path.startswith("/"):
            return f"{self.base_url}/{self.path}"
        return f"{self.base_url}{self.path}"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")
        if len(nodes) == 0:
            return []

        query_and_nodes = [
            (
                query_bundle.query_str,
                node.node.get_content(metadata_mode=MetadataMode.EMBED),
            )
            for node in nodes
        ]

        with self.callback_manager.event(
            CBEventType.RERANKING,
            payload={
                EventPayload.NODES: nodes,
                EventPayload.MODEL_NAME: self.model,
                EventPayload.QUERY_STR: query_bundle.query_str,
                EventPayload.TOP_K: self.top_k,
            },
        ) as event:
            scores = self._api_call(query_and_nodes)

            assert len(scores) == len(nodes)

            for node, score in zip(nodes, scores):
                if self.keep_retrieval_score:
                    # keep the retrieval score in metadata
                    node.node.metadata["retrieval_score"] = node.score
                node.score = score

            new_nodes = sorted(nodes, key=lambda x: -x.score if x.score else 0)[
                : self.top_k
            ]
            event.on_end(payload={EventPayload.NODES: new_nodes})

        return new_nodes

    @retry(exceptions=(httpx.RequestError), tries=3, delay=0.2)
    def _api_call(self, query_and_nodes: List[Tuple[str, str]]) -> List[float]:
        with httpx.Client() as client:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            response = client.post(
                self.endpoint,
                json={
                    "input": query_and_nodes,
                    "model": self.model,
                },
                headers=headers,
                timeout=self.timeout,
            )

            response.raise_for_status()
            return response.json()["data"]
