from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass

import requests


@dataclass(frozen=True, slots=True)
class RerankResult:
    index: int
    score: float


class JinaReranker:
    """Client for rag-bench's local FastAPI Jina reranker endpoint."""

    def __init__(
        self,
        endpoint_url: str,
        model_name: str = "jina-reranker-v3.5",
        top_n: int = 5,
        timeout_seconds: float = 15.0,
    ) -> None:
        self.endpoint_url = endpoint_url
        self.model_name = model_name
        self.top_n = top_n
        self.timeout_seconds = timeout_seconds

    async def rerank(self, query: str, documents: list[str], top_n: int) -> list[RerankResult]:
        if not documents:
            return []
        return await asyncio.to_thread(self._rerank_sync, query, documents, top_n)

    def _rerank_sync(
        self, query: str, documents: Sequence[str], top_n: int
    ) -> list[RerankResult]:
        response = requests.post(
            self.endpoint_url,
            headers={"Content-Type": "application/json"},
            json={
                "query": query,
                "documents": list(documents),
                "top_n": min(top_n, len(documents)),
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return [
            RerankResult(index=int(item["index"]), score=float(item["relevance_score"]))
            for item in response.json().get("results", [])
        ][:top_n]
