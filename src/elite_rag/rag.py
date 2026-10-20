"""Grounded RAG with single-query tri-vector retrieval and BGE reranking."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from openai import OpenAI
from qdrant_client import QdrantClient, models

from elite_rag.config import Settings
from elite_rag.embedding import HybridEmbedder
from elite_rag.multi import multi_settings, validate_multi_collection

LOGGER = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = (
    "You are a helpful, precise AI assistant. Answer the user's query using only facts "
    "explicitly present in Retrieved Context. Do not cite source labels such as [Doc 1]. "
    "Do not use outside knowledge or make assumptions. If the context does not contain "
    "enough information to answer, reply exactly: 'I cannot find the answer in the provided "
    "documents.' Keep the answer professional, concise, and direct."
)


def _chat_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key, timeout=180)


def _query_text(candidate: dict[str, Any]) -> str:
    section_path = candidate.get("section_path")
    section = (
        " > ".join(str(part) for part in section_path)
        if isinstance(section_path, list)
        else ""
    )
    return "\n".join(
        part for part in (
            f"Title: {candidate.get('title')}" if candidate.get("title") else "",
            f"Section: {section}" if section else "",
            str(candidate.get("text") or ""),
        ) if part
    )


class TriVectorRetriever:
    """Search dense, BGE learned sparse, and Qdrant BM25 with server-side RRF."""

    def __init__(self, settings: Settings) -> None:
        self.settings = multi_settings(settings)
        self.embedder = HybridEmbedder(
            self.settings.embedding_model,
            self.settings.embedding_dimension,
            self.settings.embedding_base_url,
            self.settings.embedding_api_key,
        )
        self.client = QdrantClient(
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key or None,
            timeout=180,
            cloud_inference=True,
        )
        validate_multi_collection(self.client, self.settings)

    def close(self) -> None:
        self.client.close()

    def search(self, query: str, limit: int | None = None) -> list[dict[str, Any]]:
        number = limit or self.settings.retrieval_number
        vector = self.embedder.embed_queries([query])[0]
        source_filter = models.Filter(
            must=[models.FieldCondition(
                key="ingestion_ready", match=models.MatchValue(value=True)
            )]
        )
        result = self.client.query_points(
            collection_name=self.settings.qdrant_collection,
            prefetch=[
                models.Prefetch(
                    query=vector.dense,
                    using=self.settings.qdrant_dense_vector_name,
                    filter=source_filter,
                    limit=number,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=vector.sparse_indices, values=vector.sparse_values
                    ),
                    using=self.settings.qdrant_sparse_vector_name,
                    filter=source_filter,
                    limit=number,
                ),
                models.Prefetch(
                    query=models.Document(text=query, model=self.settings.bm25_model),
                    using=self.settings.qdrant_bm25_vector_name,
                    filter=source_filter,
                    limit=number,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=number,
            with_payload=True,
            with_vectors=False,
        ).points
        found: list[dict[str, Any]] = []
        for rank, hit in enumerate(result, start=1):
            payload = dict(hit.payload or {})
            text = payload.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            found.append({
                **payload,
                "passage_id": str(payload.get("passage_id") or hit.id),
                "document_id": str(
                    payload.get("document_id") or payload.get("parent_document_id") or ""
                ),
                "rank": rank,
                "score": float(hit.score or 0.0),
                "text": text,
            })
        return found


class RagService:
    """Single-query tri-vector retrieval, BGE reranking, and grounded generation."""

    def __init__(self, settings: Settings, retriever: TriVectorRetriever | None = None) -> None:
        self.settings = multi_settings(settings)
        self.retriever = retriever or TriVectorRetriever(self.settings)

    def close(self) -> None:
        self.retriever.close()

    def retrieve(self, question: str) -> tuple[list[dict[str, Any]], list[str]]:
        # Search exactly what the caller asked; there is no query expansion.
        candidates = self.retriever.search(question)

        warnings: list[str] = []
        if not candidates:
            return [], warnings
        try:
            ranked = self.rerank(question, candidates)
        except Exception as exc:
            LOGGER.exception("bge_reranking_failed")
            warnings.append(f"BGE reranking failed; retaining RRF order: {exc}")
            ranked = candidates
        return ranked[: self.settings.rerank_top_n], warnings

    def rerank(
        self, question: str, candidates: Sequence[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        texts = [_query_text(item) for item in candidates]
        body = json.dumps({
            "query": question,
            "documents": texts,
            "top_n": len(texts),
        }).encode("utf-8")
        request = Request(
            self.settings.rerank_url,
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.settings.rerank_timeout_seconds) as response:
                payload = json.load(response)
        except HTTPError as exc:
            raise RuntimeError(
                f"BGE reranker returned HTTP {exc.code}: "
                f"{exc.read().decode('utf-8', errors='replace')}"
            ) from exc
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(f"BGE reranker request failed: {exc}") from exc
        items = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(items, list) or len(items) != len(candidates):
            raise ValueError("BGE reranker must return one ranked result per candidate")
        ranked: list[dict[str, Any]] = []
        seen: set[int] = set()
        for rank, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                raise ValueError("BGE reranker returned a malformed result")
            index, score = item.get("index"), item.get("relevance_score")
            if type(index) is not int or not 0 <= index < len(candidates) or index in seen:
                raise ValueError(f"BGE reranker returned an invalid candidate index: {index!r}")
            if not isinstance(score, (int, float)):
                raise ValueError("BGE reranker returned a non-numeric score")
            seen.add(index)
            ranked.append({**candidates[index], "rerank_rank": rank, "rerank_score": float(score)})
        if len(seen) != len(candidates):
            raise ValueError("BGE reranker omitted one or more candidates")
        return ranked

    def answer(self, question: str) -> dict[str, Any]:
        sources, warnings = self.retrieve(question)
        if not sources:
            return {
                "question": question,
                "answer": "I cannot find the answer in the provided documents.",
                "contexts": [],
                "sources": [],
                "warnings": warnings,
            }
        context = "\n\n".join(
            f"---\nSource ID: [Doc {index}]\nContent: {item['text']}\n---"
            for index, item in enumerate(sources, start=1)
        )
        response = _chat_client(
            self.settings.openai_local_endpoint, self.settings.rag_api_key
        ).chat.completions.create(
            model=self.settings.rag_model,
            temperature=self.settings.rag_temperature,
            max_tokens=self.settings.rag_max_tokens,
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"[Retrieved Context]\n{context}\n\n[User Query]\n{question}",
                },
            ],
        )
        answer = (response.choices[0].message.content or "").strip()
        if not answer:
            raise RuntimeError("RAG model returned an empty answer")
        return {
            "question": question,
            "answer": answer,
            "contexts": [item["text"] for item in sources],
            "sources": [
                {
                    "passage_id": item["passage_id"],
                    "document_id": item["document_id"],
                    "title": item.get("title"),
                    "section_name": item.get("section_name"),
                    "section_path": item.get("section_path"),
                    "source": item.get("source"),
                    "rerank_score": item.get("rerank_score"),
                }
                for item in sources
            ],
            "warnings": warnings,
        }
