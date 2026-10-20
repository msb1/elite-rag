from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from elite_rag.analyze import run_analysis
from elite_rag.benchmark import run_benchmark
from elite_rag.config import get_settings
from elite_rag.documents import SourceDocument
from elite_rag.eval import run_evaluation
from elite_rag.evaluation import RagEvaluator
from elite_rag.ingestion import IngestionReport, IngestionService
from elite_rag.jobs import IngestionJobs
from elite_rag.logging_config import configure_logging
from elite_rag.rag import RagService
from elite_rag.sources import ObjectRef, RustFSReader
from elite_rag.store import CollectionSchemaError

LOGGER = logging.getLogger(__name__)


class CollectionInitializeRequest(BaseModel):
    recreate: bool = Field(
        default=False, description="Delete and recreate the configured collection"
    )


class CollectionResponse(BaseModel):
    collection_name: str
    created: bool
    recreated: bool
    indexed_fields: list[str]


class IngestionReportResponse(BaseModel):
    documents_seen: int = 0
    documents_skipped_unchanged: int = 0
    documents_ingested: int = 0
    documents_skipped_empty: int = 0
    documents_skipped_failed: int = 0
    parents_created: int = 0
    expected_passages: int = 0
    verified_passages: int = 0
    children_upserted: int = 0
    batches_upserted: int = 0
    documents_replaced: int = 0


class IngestionJobResponse(BaseModel):
    job_id: str
    status: str
    bucket: str
    prefix: str
    submitted_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    report: IngestionReportResponse | None = None
    error: str | None = None
    work_items_total: int = 0
    work_items_completed: int = 0
    work_items_retryable: int = 0
    work_items_permanent_failed: int = 0


class RustFSPrefixIngestRequest(BaseModel):
    bucket: str = Field(min_length=1)
    prefix: str = Field(min_length=1)
    source: str | None = None
    limit: int | None = Field(default=None, ge=1)
    replace_existing: bool = False


class RustFSObjectIngestRequest(BaseModel):
    bucket: str = Field(min_length=1)
    key: str = Field(min_length=1)
    source: str | None = None
    replace_existing: bool = False


class DirectDocumentIngestRequest(BaseModel):
    document_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    source: str = Field(default="manual", min_length=1)
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    replace_existing: bool = False


class RagRequest(BaseModel):
    question: str = Field(min_length=1, max_length=20_000)


class EvaluationRequest(BaseModel):
    question: str = Field(min_length=1)
    answer: str
    contexts: list[str] = Field(default_factory=list)
    ground_truth: str = ""


class BatchRequest(BaseModel):
    limit: int | None = Field(default=None, ge=1)


@lru_cache(maxsize=1)
def _service() -> IngestionService:
    return IngestionService(get_settings())


@lru_cache(maxsize=1)
def _jobs() -> IngestionJobs:
    return IngestionJobs(get_settings(), _service())


@lru_cache(maxsize=1)
def _rag_service() -> RagService:
    return RagService(get_settings())


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_file, settings.log_level)
    try:
        manager = await asyncio.to_thread(_jobs)
        collection = await asyncio.to_thread(_service().store.ensure_collection)
        if collection.created:
            await asyncio.to_thread(manager.clear_current_collection)
        manager.resume_unfinished()
    except Exception:
        LOGGER.exception("durable_ingestion_startup_failed")
        raise
    yield


app = FastAPI(
    title="Elite RAG API",
    description=(
        "Hierarchical ingestion, tri-vector retrieval, BGE reranking, grounded RAG, "
        "and custom evaluation."
    ),
    version="3.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def protect_api(request: Request, call_next):
    key = get_settings().api_key
    if key and request.url.path.startswith("/v1/"):
        authorization = request.headers.get("authorization", "")
        supplied = authorization.removeprefix("Bearer ")
        if not secrets.compare_digest(supplied, key):
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    return await call_next(request)


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/retrieve", tags=["retrieval"])
async def retrieve(request: RagRequest) -> dict[str, Any]:
    try:
        passages, warnings = await asyncio.to_thread(
            _rag_service().retrieve, request.question
        )
        return {"question": request.question, "passages": passages,
                "warnings": warnings}
    except Exception as exc:
        LOGGER.exception("retrieval_failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/v1/rag", tags=["rag"])
async def rag(request: RagRequest) -> dict[str, Any]:
    try:
        return await asyncio.to_thread(_rag_service().answer, request.question)
    except Exception as exc:
        LOGGER.exception("rag_answer_failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/v1/evaluate", tags=["evaluation"])
async def evaluate(request: EvaluationRequest) -> dict[str, float]:
    try:
        evaluator = RagEvaluator(get_settings())
        return await asyncio.to_thread(
            evaluator.evaluate,
            {
                "question": request.question,
                "answer": request.answer,
                "contexts": request.contexts,
                "ground_truth": request.ground_truth,
            },
        )
    except Exception as exc:
        LOGGER.exception("evaluation_failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/v1/eval", tags=["evaluation"])
async def evaluate_dataset(request: BatchRequest) -> dict[str, Any]:
    settings = get_settings()
    try:
        count = await asyncio.to_thread(run_evaluation, settings, limit=request.limit)
        return {"status": "completed", "exit_code": count,
                "output_file": settings.eval_output_file}
    except Exception as exc:
        LOGGER.exception("dataset_evaluation_failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/v1/benchmark", tags=["benchmark"])
async def run_dataset_benchmark(request: BatchRequest) -> dict[str, Any]:
    settings = get_settings()
    try:
        status_code = await asyncio.to_thread(run_benchmark, settings, limit=request.limit)
        return {"status": "completed", "exit_code": status_code,
                "output_file": settings.dataset_file}
    except Exception as exc:
        LOGGER.exception("benchmark_run_failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/v1/analyze", tags=["evaluation"])
async def analyze_dataset() -> dict[str, Any]:
    settings = get_settings()
    try:
        summary = await asyncio.to_thread(run_analysis, settings)
        return summary
    except Exception as exc:
        LOGGER.exception("analysis_failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/v1/collections/initialize", response_model=CollectionResponse, tags=["collections"])
async def initialize_collection(request: CollectionInitializeRequest) -> CollectionResponse:
    try:
        report = await asyncio.to_thread(_service().store.ensure_collection, request.recreate)
        if report.created:
            await asyncio.to_thread(_jobs().clear_current_collection)
        return CollectionResponse(
            collection_name=report.collection_name,
            created=report.created,
            recreated=report.recreated,
            indexed_fields=list(report.indexed_fields),
        )
    except CollectionSchemaError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("collection_initialize_failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post(
    "/v1/ingest/rustfs/prefix",
    response_model=IngestionJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["ingestion"],
)
async def ingest_rustfs_prefix(request: RustFSPrefixIngestRequest) -> IngestionJobResponse:
    try:
        result = _jobs().submit(
            request.bucket, request.prefix, request.source, request.limit, request.replace_existing
        )
        return IngestionJobResponse.model_validate(result)
    except Exception as exc:
        LOGGER.exception("prefix_ingestion_submit_failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/v1/ingest/jobs/{job_id}", response_model=IngestionJobResponse, tags=["ingestion"])
async def get_ingestion_job(job_id: str) -> IngestionJobResponse:
    result = await asyncio.to_thread(_jobs().get, job_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Ingestion job {job_id!r} was not found")
    return IngestionJobResponse.model_validate(result)


@app.post(
    "/v1/ingest/rustfs/object",
    response_model=IngestionReportResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["ingestion"],
)
async def ingest_rustfs_object(request: RustFSObjectIngestRequest) -> IngestionReportResponse:
    try:
        return await asyncio.to_thread(_ingest_object, request)
    except Exception as exc:
        LOGGER.exception("rustfs_object_ingestion_failed key=%s", request.key)
        raise HTTPException(status_code=502, detail=str(exc)) from exc


def _ingest_object(request: RustFSObjectIngestRequest) -> IngestionReportResponse:
    reader = RustFSReader(get_settings())
    ref = reader.head(request.bucket, request.key, request.source)
    if _jobs().is_current(ref) and not request.replace_existing:
        return IngestionReportResponse()
    document = reader.read(ref)
    report = _service().ingest(document, request.replace_existing)
    _jobs().record_current(ref, document.document_id)
    return IngestionReportResponse(**asdict(report))


@app.post(
    "/v1/ingest/document",
    response_model=IngestionReportResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["ingestion"],
)
async def ingest_document(request: DirectDocumentIngestRequest) -> IngestionReportResponse:
    try:
        return await asyncio.to_thread(_ingest_document, request)
    except Exception as exc:
        LOGGER.exception("direct_document_ingestion_failed document_id=%s", request.document_id)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _ingest_document(request: DirectDocumentIngestRequest) -> IngestionReportResponse:
    document = SourceDocument(
        document_id=request.document_id,
        source=request.source.lower(),
        title=request.title or request.document_id,
        fields={"content": request.content},
        metadata=request.metadata,
    )
    fingerprint = hashlib.sha256(
        json.dumps(request.model_dump(), sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    ref = ObjectRef(
        "__direct__", request.document_id, fingerprint, len(request.content), request.source
    )
    if _jobs().is_current(ref) and not request.replace_existing:
        return IngestionReportResponse()
    report: IngestionReport = _service().ingest(document, request.replace_existing)
    _jobs().record_current(ref, document.document_id)
    return IngestionReportResponse(**asdict(report))
