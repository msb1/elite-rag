"""BGE reranking and BGE-M3 dense plus lexical sparse embeddings."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

import psutil
import torch
import uvicorn
from dotenv import dotenv_values
from fastapi import FastAPI, HTTPException
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ENV = dotenv_values(".env")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_COMPILE_BACKEND"] = "eager"

LOGGER = logging.getLogger("remote_models")
RERANK_MODEL_NAME = ENV.get("RERANK_MODEL") or "BAAI/bge-reranker-v2-m3"
RERANK_DEVICE = ENV.get("RERANKER_DEVICE") or "cuda"
RERANK_MAX_LENGTH = int(ENV.get("RERANK_MAX_LENGTH") or "512")
RERANK_BATCH_SIZE = int(ENV.get("RERANK_BATCH_SIZE") or "8")
BGE_MODEL_NAME = ENV.get("BGE_MODEL_NAME") or "BAAI/bge-m3"
BGE_MODEL_ID = ENV.get("EMBEDDING_MODEL") or "bge-m3"
BGE_DEVICE = ENV.get("BGE_EMBEDDING_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
BGE_MAX_LENGTH = int(ENV.get("BGE_EMBEDDING_MAX_LENGTH") or "1024")
BGE_MAX_INPUTS = int(ENV.get("BGE_EMBEDDING_MAX_INPUTS") or "64")
BGE_MAX_TEXT_CHARACTERS = int(ENV.get("BGE_EMBEDDING_MAX_TEXT_CHARACTERS") or "100000")
reranker_tokenizer: Any = None
reranker_model: Any = None
bge_model: SentenceTransformer | None = None
bge_sparse_head: torch.nn.Linear | None = None
reranker_lock = threading.Lock()
bge_lock = threading.Lock()


class RerankRequest(BaseModel):
    query: str = Field(..., description="The user query string.")
    documents: list[str] = Field(..., description="Document text contents to score.")
    top_n: int = Field(default=5, ge=0, description="Number of sorted items to return.")


class RerankResultItem(BaseModel):
    index: int
    relevance_score: float


class RerankResponse(BaseModel):
    results: list[RerankResultItem]


class EmbeddingRequest(BaseModel):
    model: str = Field(..., description="Embedding model ID.")
    input: str | list[str] = Field(..., description="Text or texts to embed.")
    # Accepted for compatibility with existing clients; BGE-M3 uses raw text for both roles.
    input_type: Literal["query", "document"] | None = Field(
        default=None,
        description="Query or document role; BGE-M3 requires no task prefix.",
    )


class SparseVectorData(BaseModel):
    indices: list[int]
    values: list[float]


class EmbeddingItem(BaseModel):
    object: Literal["embedding"] = "embedding"
    embedding: list[float]
    sparse_vector: SparseVectorData
    index: int


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[EmbeddingItem]
    model: str
    usage: EmbeddingUsage


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "local"


class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    reranker_model: str
    reranker: dict[str, Any]
    embedding: dict[str, Any]
    rss_bytes: int
    cuda_available: bool
    cuda_allocated_bytes: int | None = None
    cuda_reserved_bytes: int | None = None


def _load_reranker() -> tuple[Any, Any]:
    LOGGER.info("loading_reranker model=%s device=%s", RERANK_MODEL_NAME, RERANK_DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        RERANK_MODEL_NAME,
        dtype="auto",
    ).to(RERANK_DEVICE)
    model.eval()
    return tokenizer, model


def _load_bge_embedder() -> tuple[SentenceTransformer, torch.nn.Linear]:
    if BGE_MAX_INPUTS < 1:
        raise ValueError("BGE_EMBEDDING_MAX_INPUTS must be positive")
    if not 8 <= BGE_MAX_LENGTH <= 8192:
        raise ValueError("BGE_EMBEDDING_MAX_LENGTH must be between 8 and 8192")
    if BGE_DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("BGE_EMBEDDING_DEVICE requests CUDA, but CUDA is unavailable")
    LOGGER.info(
        "loading_bge_embedding model=%s device=%s batch_size=1 max_length=%s",
        BGE_MODEL_NAME,
        BGE_DEVICE,
        BGE_MAX_LENGTH,
    )
    model = SentenceTransformer(BGE_MODEL_NAME, device=BGE_DEVICE)
    model.max_seq_length = BGE_MAX_LENGTH
    model.eval()
    dimension = model.get_sentence_embedding_dimension()
    if dimension != 1024:
        raise RuntimeError(f"BGE-M3 should have 1024 dimensions; got {dimension}")
    # The Sentence Transformers model exposes dense and token states. BAAI publishes
    # this trained lexical projection separately as sparse_linear.pt.
    head = torch.nn.Linear(dimension, 1)
    weights_path = hf_hub_download(repo_id=BGE_MODEL_NAME, filename="sparse_linear.pt")
    head.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    head.to(BGE_DEVICE).eval()
    return model, head


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    global reranker_tokenizer, reranker_model, bge_model, bge_sparse_head
    if RERANK_MAX_LENGTH < 1 or RERANK_BATCH_SIZE < 1:
        raise ValueError("RERANK_MAX_LENGTH and RERANK_BATCH_SIZE must be positive")
    # Load sequentially so both model loads do not spike host/GPU memory at once.
    reranker_tokenizer, reranker_model = await asyncio.to_thread(_load_reranker)
    bge_model, bge_sparse_head = await asyncio.to_thread(_load_bge_embedder)
    LOGGER.info(
        "remote_models_ready reranker=%s embedding=%s embedding_device=%s embedding_max_length=%s",
        RERANK_MODEL_NAME,
        BGE_MODEL_NAME,
        BGE_DEVICE,
        BGE_MAX_LENGTH,
    )
    try:
        yield
    finally:
        reranker_tokenizer = reranker_model = bge_model = bge_sparse_head = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


app = FastAPI(
    title="Elite RAG Remote BGE Reranker and BGE-M3 Embeddings",
    description="BGE reranking and BGE-M3 dense and lexical sparse embeddings.",
    version="1.0.0",
    lifespan=lifespan,
)


def _rerank(query: str, documents: list[str]) -> list[dict[str, Any]]:
    if reranker_model is None or reranker_tokenizer is None:
        raise RuntimeError("reranker is not loaded")
    all_results: list[dict[str, Any]] = []
    for start in range(0, len(documents), RERANK_BATCH_SIZE):
        batch_docs = documents[start : start + RERANK_BATCH_SIZE]
        pairs = [[query, document] for document in batch_docs]
        inputs = reranker_tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=RERANK_MAX_LENGTH,
            return_tensors="pt",
        ).to(RERANK_DEVICE)
        with torch.inference_mode():
            scores = reranker_model(**inputs, return_dict=True).logits.view(-1).float()
        for offset, score in enumerate(scores.cpu().tolist()):
            all_results.append(
                {
                    "index": start + offset,
                    "relevance_score": float(score),
                }
            )
    return sorted(all_results, key=lambda item: item["relevance_score"], reverse=True)


def _embed(texts: list[str]) -> tuple[list[tuple[list[float], SparseVectorData]], int]:
    if bge_model is None or bge_sparse_head is None:
        raise RuntimeError("BGE-M3 embedding model is not loaded")
    if len(texts) > BGE_MAX_INPUTS:
        raise ValueError(f"At most {BGE_MAX_INPUTS} texts are allowed per request")
    if any(len(text) > BGE_MAX_TEXT_CHARACTERS for text in texts):
        raise ValueError(f"Each text is limited to {BGE_MAX_TEXT_CHARACTERS} characters")

    vectors: list[tuple[list[float], SparseVectorData]] = []
    total_tokens = 0
    started = time.perf_counter()
    tokenizer = bge_model.tokenizer
    special_ids = {
        token_id
        for token_id in (
            tokenizer.cls_token_id,
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
            tokenizer.unk_token_id,
        )
        if token_id is not None
    }
    with bge_lock, torch.inference_mode():
        for text in texts:
            features = bge_model.encode(
                text,
                batch_size=1,
                output_value=None,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
            dense = features["sentence_embedding"].float().cpu().tolist()
            token_states = features["token_embeddings"].to(BGE_DEVICE)
            weights = torch.relu(bge_sparse_head(token_states)).squeeze(-1)
            token_ids = features["input_ids"].tolist()
            attention = features["attention_mask"].tolist()
            total_tokens += sum(attention)
            lexical: dict[int, float] = {}
            for token_id, present, weight in zip(
                token_ids, attention, weights.float().cpu().tolist(), strict=True
            ):
                token_id = int(token_id)
                if not present or token_id in special_ids or weight <= 0:
                    continue
                lexical[token_id] = max(lexical.get(token_id, 0.0), float(weight))
            indices = sorted(lexical)
            sparse = SparseVectorData(indices=indices, values=[lexical[i] for i in indices])
            vectors.append((dense, sparse))

    if any(len(dense) != 1024 for dense, _ in vectors):
        raise RuntimeError("BGE-M3 should return 1024-dimensional dense vectors")
    LOGGER.info(
        "bge_embedding_completed texts=%d tokens=%d duration_ms=%.1f",
        len(texts),
        total_tokens,
        (time.perf_counter() - started) * 1000,
    )
    return vectors, total_tokens


@app.post("/v1/rerank", response_model=RerankResponse, tags=["reranking"])
async def rerank_endpoint(payload: RerankRequest) -> RerankResponse:
    if not payload.documents or payload.top_n == 0:
        return RerankResponse(results=[])
    try:
        results = await asyncio.to_thread(
            _rerank_with_lock, payload.query, payload.documents
        )
    except Exception as exc:
        LOGGER.exception("rerank_failed documents=%s", len(payload.documents))
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return RerankResponse(results=[RerankResultItem(**item) for item in results[: payload.top_n]])


def _rerank_with_lock(query: str, documents: list[str]) -> list[dict[str, Any]]:
    with reranker_lock:
        return _rerank(query, documents)


@app.post("/v1/embeddings", response_model=EmbeddingResponse, tags=["embeddings"])
async def embeddings_endpoint(payload: EmbeddingRequest) -> EmbeddingResponse:
    texts = [payload.input] if isinstance(payload.input, str) else payload.input
    if payload.model not in {BGE_MODEL_ID, BGE_MODEL_NAME}:
        raise HTTPException(
            status_code=404,
            detail=f"This service provides {BGE_MODEL_ID!r} ({BGE_MODEL_NAME})",
        )
    if not texts:
        return EmbeddingResponse(
            data=[], model=BGE_MODEL_ID, usage=EmbeddingUsage(prompt_tokens=0, total_tokens=0)
        )
    try:
        vectors, token_count = await asyncio.to_thread(_embed, texts)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("bge_embedding_failed texts=%s", len(texts))
        raise HTTPException(
            status_code=503,
            detail={"error_type": type(exc).__name__, "reason": str(exc)},
        ) from exc
    return EmbeddingResponse(
        data=[
            EmbeddingItem(index=index, embedding=dense, sparse_vector=sparse)
            for index, (dense, sparse) in enumerate(vectors)
        ],
        model=BGE_MODEL_ID,
        usage=EmbeddingUsage(prompt_tokens=token_count, total_tokens=token_count),
    )


@app.get("/v1/models", response_model=ModelListResponse, tags=["models"])
async def list_models() -> ModelListResponse:
    return ModelListResponse(data=[ModelCard(id=BGE_MODEL_ID)])


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    reranker_ready = reranker_model is not None and reranker_tokenizer is not None
    embedding_ready = bge_model is not None and bge_sparse_head is not None
    cuda_available = torch.cuda.is_available()
    allocated = torch.cuda.memory_allocated() if cuda_available else None
    reserved = torch.cuda.memory_reserved() if cuda_available else None
    return HealthResponse(
        status="ok" if reranker_ready and embedding_ready else "degraded",
        reranker_model=RERANK_MODEL_NAME,
        reranker={"loaded": reranker_ready, "model": RERANK_MODEL_NAME, "device": RERANK_DEVICE},
        embedding={
            "loaded": embedding_ready,
            "model": BGE_MODEL_NAME,
            "api_model_id": BGE_MODEL_ID,
            "device": BGE_DEVICE,
            "dimension": 1024,
            "max_length": BGE_MAX_LENGTH,
            "batch_size": 1,
            "sparse": True,
        },
        rss_bytes=psutil.Process().memory_info().rss,
        cuda_available=cuda_available,
        cuda_allocated_bytes=allocated,
        cuda_reserved_bytes=reserved,
    )


if __name__ == "__main__":
    logging.basicConfig(level=ENV.get("LOG_LEVEL") or "INFO")
    uvicorn.run(app, host="0.0.0.0", port=int(ENV.get("PORT") or "8000"))
