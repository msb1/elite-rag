from __future__ import annotations

from types import SimpleNamespace

import httpx

from elite_rag.reranking import RerankResult
from elite_rag.retrieval import TwoPassDeterministicRetriever


class FakeEmbedder:
    def embed_query(self, query: str) -> list[float]:
        return [float(len(query))]


class FakeStore:
    def __init__(self) -> None:
        self.limits: list[int] = []

    def search(
        self, vector: list[float], query: str, search_filter: object, limit: int
    ) -> list[object]:
        self.limits.append(limit)
        suffix = len(self.limits)
        return [
            SimpleNamespace(
                payload={
                    "parent_id": f"p{suffix}",
                    "document_id": f"d{suffix}",
                    "parent_context": f"context {suffix}",
                    "metadata_filters": {"source": "jira"},
                },
                score=0.9,
            )
        ]


class FakeIntent:
    def extract(self, query: str) -> None:
        return None


class LowThenHighReranker:
    def __init__(self) -> None:
        self.calls = 0

    async def rerank(self, query: str, documents: list[str], top_n: int) -> list[RerankResult]:
        self.calls += 1
        score = 0.1 if self.calls == 1 else 0.8
        return [RerankResult(0, score)]


class BrokenReranker:
    async def rerank(self, query: str, documents: list[str], top_n: int) -> list[RerankResult]:
        raise httpx.ConnectError("offline")


async def test_low_confidence_runs_exactly_one_corrective_pass() -> None:
    store = FakeStore()
    reranker = LowThenHighReranker()
    retriever = TwoPassDeterministicRetriever(
        store,  # type: ignore[arg-type]
        FakeEmbedder(),  # type: ignore[arg-type]
        FakeIntent(),  # type: ignore[arg-type]
        reranker,
        primary_limit=25,
        fallback_limit=50,
    )
    results = await retriever.retrieve("What is the answer?")
    assert store.limits == [25, 50]
    assert reranker.calls == 2
    assert results[0].document_id == "d2"
    assert results[0].rerank_score == 0.8


async def test_reranker_outage_returns_primary_without_second_search() -> None:
    store = FakeStore()
    retriever = TwoPassDeterministicRetriever(
        store,  # type: ignore[arg-type]
        FakeEmbedder(),  # type: ignore[arg-type]
        FakeIntent(),  # type: ignore[arg-type]
        BrokenReranker(),
    )
    results = await retriever.retrieve("query")
    assert store.limits == [25]
    assert results[0].document_id == "d1"
