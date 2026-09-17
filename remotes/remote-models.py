"""Combined remote Jina reranker and miniCOIL sparse-embedding API.

Run this file on the remote model computer with ``python remote-models.py``.
It intentionally preserves the existing Jina route, ``POST /v1/rerank``, and
adds ``POST /v1/embeddings/sparse`` on the same host and port.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import psutil
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from transformers import AutoModel

LOGGER = logging.getLogger("remote_models")


@dataclass(frozen=True, slots=True)
class Settings:
    reranker_model: str
    reranker_device: str
    minicoil_model: str
    minicoil_average_document_length: float
    minicoil_max_batch_size: int
    minicoil_max_text_characters: int
    minicoil_threads: int
    log_level: str

    @classmethod
    def from_environment(cls) -> Settings:
        load_dotenv()
        return cls(
            reranker_model=os.getenv("RERANKER_MODEL", "jinaai/jina-reranker-v3.5"),
            reranker_device=os.getenv("RERANKER_DEVICE", "cuda"),
            minicoil_model=os.getenv("MINICOIL_MODEL", "Qdrant/minicoil-v1"),
            minicoil_average_document_length=_positive_float(
                "MINICOIL_AVG_DOCUMENT_LENGTH", 100.0
            ),
            minicoil_max_batch_size=_positive_int("MINICOIL_MAX_BATCH_SIZE", 8),
            minicoil_max_text_characters=_positive_int("MINICOIL_MAX_TEXT_CHARACTERS", 64_000),
            minicoil_threads=_positive_int("MINICOIL_THREADS", 2),
            log_level=os.getenv("REMOTE_MODELS_LOG_LEVEL", "INFO"),
        )


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def _positive_float(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return value


class EmbeddingMode(StrEnum):
    document = "document"
    query = "query"


class RerankRequest(BaseModel):
    """Unchanged request contract for the existing Jina reranker endpoint."""

    query: str = Field(..., description="The user query string.")
    documents: list[str] = Field(..., description="Array of document text contents to score.")
    top_n: int = Field(default=5, ge=1, description="Number of sorted items to return.")


class RerankResultItem(BaseModel):
    index: int
    relevance_score: float


class RerankResponse(BaseModel):
    results: list[RerankResultItem]


class SparseEmbeddingRequest(BaseModel):
    model: str = Field(min_length=1)
    texts: list[str] = Field(min_length=1)
    mode: EmbeddingMode


class SparseVectorResponse(BaseModel):
    indices: list[int]
    values: list[float]


class SparseEmbeddingResponse(BaseModel):
    model: str
    mode: EmbeddingMode
    vectors: list[SparseVectorResponse]


class HealthResponse(BaseModel):
    status: str
    reranker_model: str
    minicoil_model: str
    minicoil_average_document_length: float
    minicoil_max_batch_size: int
    rss_bytes: int


@dataclass(slots=True)
class RemoteModels:
    settings: Settings
    reranker: Any
    minicoil: Any
    reranker_lock: threading.Lock
    minicoil_lock: threading.Lock

    def rerank(self, query: str, documents: list[str]) -> list[dict[str, Any]]:
        with self.reranker_lock:
            return list(self.reranker.rerank(query, documents))

    def embed(
        self, texts: list[str], mode: EmbeddingMode
    ) -> list[SparseVectorResponse]:
        self._validate_texts(texts)
        started = time.perf_counter()
        with self.minicoil_lock:
            embeddings = (
                self.minicoil.query_embed(texts)
                if mode is EmbeddingMode.query
                else self.minicoil.embed(
                    texts,
                    batch_size=self.settings.minicoil_max_batch_size,
                    parallel=None,
                )
            )
            vectors = [_sparse_vector(item) for item in embeddings]
        if len(vectors) != len(texts):
            raise RuntimeError("miniCOIL returned a different number of vectors than input texts")
        LOGGER.info(
            "minicoil_embedding_completed mode=%s texts=%s duration_ms=%.1f rss_bytes=%s",
            mode.value,
            len(texts),
            (time.perf_counter() - started) * 1_000,
            _rss_bytes(),
        )
        return vectors

    def _validate_texts(self, texts: list[str]) -> None:
        if len(texts) > self.settings.minicoil_max_batch_size:
            raise ValueError(
                "At most "
                f"{self.settings.minicoil_max_batch_size} texts are allowed in one miniCOIL request"
            )
        if any(len(text) > self.settings.minicoil_max_text_characters for text in texts):
            raise ValueError(
                "Each text must contain at most "
                f"{self.settings.minicoil_max_text_characters} characters"
            )


def _load_models(settings: Settings) -> RemoteModels:
    from fastembed import SparseTextEmbedding

    LOGGER.info(
        "loading_reranker model=%s device=%s",
        settings.reranker_model,
        settings.reranker_device,
    )
    reranker = AutoModel.from_pretrained(
        settings.reranker_model,
        trust_remote_code=True,
        dtype="auto",
    ).to(settings.reranker_device)
    reranker.eval()

    LOGGER.info(
        "loading_minicoil model=%s avg_document_length=%s threads=%s",
        settings.minicoil_model,
        settings.minicoil_average_document_length,
        settings.minicoil_threads,
    )
    minicoil = SparseTextEmbedding(
        model_name=settings.minicoil_model,
        avg_len=settings.minicoil_average_document_length,
        threads=settings.minicoil_threads,
    )
    return RemoteModels(
        settings=settings,
        reranker=reranker,
        minicoil=minicoil,
        reranker_lock=threading.Lock(),
        minicoil_lock=threading.Lock(),
    )


def _sparse_vector(embedding: Any) -> SparseVectorResponse:
    return SparseVectorResponse(
        indices=[int(index) for index in embedding.indices],
        values=[float(value) for value in embedding.values],
    )


def _rss_bytes() -> int:
    return psutil.Process().memory_info().rss


def _models(request: Request) -> RemoteModels:
    models = getattr(request.app.state, "models", None)
    if models is None:
        raise HTTPException(status_code=503, detail="Remote models are not initialized")
    return models


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings.from_environment()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    app.state.models = await asyncio.to_thread(_load_models, settings)
    LOGGER.info(
        "remote_models_started reranker_model=%s minicoil_model=%s rss_bytes=%s",
        settings.reranker_model,
        settings.minicoil_model,
        _rss_bytes(),
    )
    try:
        yield
    finally:
        LOGGER.info("remote_models_stopped rss_bytes=%s", _rss_bytes())
        del app.state.models


app = FastAPI(
    title="Elite RAG Remote Models API",
    description=(
        "Jina Reranker v3.5 and bounded-memory miniCOIL sparse embeddings. "
        "The established reranker route is preserved."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health(request: Request) -> HealthResponse:
    settings = _models(request).settings
    return HealthResponse(
        status="ok",
        reranker_model=settings.reranker_model,
        minicoil_model=settings.minicoil_model,
        minicoil_average_document_length=settings.minicoil_average_document_length,
        minicoil_max_batch_size=settings.minicoil_max_batch_size,
        rss_bytes=_rss_bytes(),
    )


@app.post(
    "/v1/rerank",
    response_model=RerankResponse,
    tags=["reranking"],
    summary="Rerank documents with Jina Reranker v3.5",
)
async def rerank_endpoint(payload: RerankRequest, request: Request) -> RerankResponse:
    """Preserves the request and response format of ``services/reranker.py``."""
    if not payload.documents:
        return RerankResponse(results=[])
    try:
        results = await asyncio.to_thread(_models(request).rerank, payload.query, payload.documents)
    except Exception as exc:
        LOGGER.exception("rerank_failed documents=%s", len(payload.documents))
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # Jina returns each item's original position in ``documents``. Elite RAG uses that
    # index to recover the matching parent context, so retain it rather than emitting
    # the item's position in this already-sorted result list.
    response_items = [
        RerankResultItem(index=int(result["index"]), relevance_score=float(result["relevance_score"]))
        for result in results
    ]
    if response_items:
        LOGGER.info(
            "rerank_completed documents=%s top_index=%s top_score=%.4f rss_bytes=%s",
            len(payload.documents),
            response_items[0].index,
            response_items[0].relevance_score,
            _rss_bytes(),
        )
    return RerankResponse(results=response_items[: payload.top_n])


@app.post(
    "/v1/embeddings/sparse",
    response_model=SparseEmbeddingResponse,
    tags=["embeddings"],
    summary="Create Qdrant-compatible miniCOIL sparse vectors",
)
async def embed_sparse(
    payload: SparseEmbeddingRequest, request: Request
) -> SparseEmbeddingResponse:
    models = _models(request)
    try:
        if payload.model != models.settings.minicoil_model:
            raise ValueError(
                f"Configured model is '{models.settings.minicoil_model}', not '{payload.model}'"
            )
        vectors = await asyncio.to_thread(models.embed, payload.texts, payload.mode)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception(
            "minicoil_embedding_failed mode=%s texts=%s", payload.mode.value, len(payload.texts)
        )
        raise HTTPException(
            status_code=503,
            detail={"error_type": type(exc).__name__, "reason": str(exc)},
        ) from exc
    return SparseEmbeddingResponse(
        model=models.settings.minicoil_model,
        mode=payload.mode,
        vectors=vectors,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
