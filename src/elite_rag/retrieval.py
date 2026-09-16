from __future__ import annotations

import asyncio
import logging
from typing import Any, Protocol

import httpx
import requests

from elite_rag.embedding import EmbeddingGemma
from elite_rag.intent import DeterministicIntentRouter, fallback_query
from elite_rag.models import RetrievedParent
from elite_rag.reranking import RerankResult
from elite_rag.vector_store import QdrantVectorStore, SearchFilter

LOGGER = logging.getLogger(__name__)


class Reranker(Protocol):
    async def rerank(self, query: str, documents: list[str], top_n: int) -> list[RerankResult]: ...


class TwoPassDeterministicRetriever:
    def __init__(
        self,
        store: QdrantVectorStore,
        embedder: EmbeddingGemma,
        intent_router: DeterministicIntentRouter,
        reranker: Reranker | None,
        *,
        confidence_threshold: float = 0.35,
        primary_limit: int = 25,
        fallback_limit: int = 50,
        top_n: int = 10,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.intent_router = intent_router
        self.reranker = reranker
        self.confidence_threshold = confidence_threshold
        self.primary_limit = primary_limit
        self.fallback_limit = fallback_limit
        self.top_n = top_n

    async def retrieve(
        self,
        query: str,
        search_filter: SearchFilter | None = None,
        top_n: int | None = None,
    ) -> list[RetrievedParent]:
        active_filter = _merge_filters(self.intent_router.extract(query), search_filter)
        requested_top_n = top_n or self.top_n
        primary = await self._vector_search(query, active_filter, self.primary_limit)
        if not primary:
            return await self._fallback(query, active_filter, requested_top_n)
        if self.reranker is None:
            return primary[:requested_top_n]
        reranked = await self._rerank(query, primary, requested_top_n)
        if reranked is None:
            return primary[:requested_top_n]
        if reranked and (reranked[0].rerank_score or 0.0) >= self.confidence_threshold:
            return reranked
        return await self._fallback(query, active_filter, requested_top_n)

    async def _fallback(
        self, original_query: str, active_filter: SearchFilter | None, top_n: int
    ) -> list[RetrievedParent]:
        optimized = fallback_query(original_query)
        candidates = await self._vector_search(optimized, active_filter, self.fallback_limit)
        if not candidates:
            return []
        reranked = await self._rerank(optimized, candidates, top_n)
        return reranked or candidates[:top_n]

    async def _vector_search(
        self, query: str, active_filter: SearchFilter | None, limit: int
    ) -> list[RetrievedParent]:
        vector = await asyncio.to_thread(self.embedder.embed_query, query)
        hits = await asyncio.to_thread(self.store.search, vector, query, active_filter, limit)
        return deduplicate_parents(hits)

    async def _rerank(
        self, query: str, candidates: list[RetrievedParent], top_n: int
    ) -> list[RetrievedParent] | None:
        if self.reranker is None:
            return None
        try:
            results = await self.reranker.rerank(
                query, [candidate.text for candidate in candidates], top_n
            )
        except (httpx.HTTPError, requests.RequestException, KeyError, TypeError, ValueError) as exc:
            LOGGER.warning("Jina reranking failed; using vector order: %s", exc)
            return None
        ranked: list[RetrievedParent] = []
        for result in results:
            if 0 <= result.index < len(candidates):
                candidate = candidates[result.index]
                candidate.rerank_score = result.score
                ranked.append(candidate)
        return ranked


def deduplicate_parents(hits: list[Any]) -> list[RetrievedParent]:
    parents: list[RetrievedParent] = []
    seen: set[str] = set()
    for hit in hits:
        payload = hit.payload or {}
        parent_id = payload.get("parent_id")
        if not parent_id or parent_id in seen:
            continue
        seen.add(parent_id)
        parents.append(
            RetrievedParent(
                parent_id=str(parent_id),
                document_id=str(payload.get("document_id", parent_id)),
                text=str(payload.get("parent_context", "")),
                metadata=dict(payload.get("metadata_filters", {})),
                vector_score=float(hit.score) if getattr(hit, "score", None) is not None else None,
            )
        )
    return parents


def _merge_filters(
    inferred: SearchFilter | None, requested: SearchFilter | None
) -> SearchFilter | None:
    if inferred is None:
        return requested
    if requested is None:
        return inferred
    return SearchFilter(
        sources=requested.sources or inferred.sources,
        projects=requested.projects or inferred.projects,
    )
