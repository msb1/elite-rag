from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Request, status
from starlette.responses import Response

from elite_rag.api_models import (
    CollectionInitializeRequest,
    CollectionResponse,
    DirectDocumentIngestRequest,
    IngestionJobResponse,
    IngestionReportResponse,
    RagRequest,
    RagResponse,
    RustFSObjectIngestRequest,
    RustFSPrefixIngestRequest,
    ScoringRequest,
    ScoringResponse,
)
from elite_rag.config import get_settings
from elite_rag.evaluation import LlamaEvaluator
from elite_rag.ingestion_ledger import IngestionLedger
from elite_rag.jobs import IngestionJobManager
from elite_rag.logging_config import configure_logging
from elite_rag.runtime import build_runtime
from elite_rag.service import RAGService
from elite_rag.vector_store import CollectionSchemaError

LOGGER = logging.getLogger(__name__)

app = FastAPI(
    title="Elite RAG Server API",
    description=(
        "Production API for RustFS ingestion, retrieval-augmented generation, and "
        "Llama 3.1 evaluation. Interactive documentation is available at /docs."
    ),
    version="1.0.0",
)


@lru_cache(maxsize=1)
def get_service() -> RAGService:
    settings = get_settings()
    return RAGService(build_runtime(settings), settings, LlamaEvaluator(settings))


@lru_cache(maxsize=1)
def get_job_manager() -> IngestionJobManager:
    settings = get_settings()
    if not settings.ingestion_database_url:
        raise RuntimeError("INGESTION_DATABASE_URL must be configured for durable ingestion")
    return IngestionJobManager(
        IngestionLedger(settings.ingestion_database_url, settings.ingestion_lease_seconds)
    )


async def ensure_ledger_schema() -> IngestionJobManager:
    """Initialize the durable ledger only in response to an explicit API request."""
    manager = get_job_manager()
    await asyncio.to_thread(manager.ledger.ensure_schema)
    return manager


@app.middleware("http")
async def log_request(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    settings = get_settings()
    configure_logging(settings.log_file, settings.log_level)
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        LOGGER.exception("http_request_failed method=%s path=%s", request.method, request.url.path)
        raise
    LOGGER.info(
        "http_request_completed method=%s path=%s status=%s duration_ms=%.1f",
        request.method,
        request.url.path,
        response.status_code,
        (time.perf_counter() - started) * 1000,
    )
    return response


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/v1/collections/initialize",
    response_model=CollectionResponse,
    tags=["collections"],
    summary="Create or verify the Elite RAG Qdrant collection",
)
async def initialize_collection(request: CollectionInitializeRequest) -> CollectionResponse:
    try:
        manager = await ensure_ledger_schema()
        report = await get_service().initialize_collection(request.recreate)
        if report.created or report.recreated:
            await asyncio.to_thread(manager.ledger.clear)
            LOGGER.info(
                "ingestion_ledger_cleared collection=%s recreated=%s created=%s",
                report.collection_name,
                report.recreated,
                report.created,
            )
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
        raise _service_error(503, "Qdrant collection operation failed", exc) from exc


@app.post(
    "/v1/ingest/rustfs/prefix",
    response_model=IngestionJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["ingestion"],
    summary="Recursively ingest all documents under a RustFS prefix",
)
async def ingest_rustfs_prefix(request: RustFSPrefixIngestRequest) -> IngestionJobResponse:
    settings = get_settings()
    manager = await ensure_ledger_schema()
    job = await manager.submit(
        bucket=request.bucket,
        prefix=request.prefix,
        source=request.source,
        limit=request.limit,
        replace_existing=request.replace_existing,
        collection_name=settings.qdrant_collection,
        pipeline_version=settings.ingestion_pipeline_version,
        operation=lambda job_id: get_service().ingest_durable_prefix_run(job_id, manager.ledger),
    )
    return IngestionJobResponse.model_validate(job.response())


@app.get(
    "/v1/ingest/jobs/{job_id}",
    response_model=IngestionJobResponse,
    tags=["ingestion"],
    summary="Get RustFS-prefix ingestion progress and failures",
)
async def get_ingestion_job(job_id: str) -> IngestionJobResponse:
    job = (await ensure_ledger_schema()).get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Ingestion job '{job_id}' was not found")
    return IngestionJobResponse.model_validate(job.response())


@app.post(
    "/v1/ingest/rustfs/object",
    response_model=IngestionReportResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["ingestion"],
    summary="Ingest one RustFS object by bucket and key",
)
async def ingest_rustfs_object(request: RustFSObjectIngestRequest) -> IngestionReportResponse:
    try:
        return await get_service().ingest_object(request, (await ensure_ledger_schema()).ledger)
    except Exception as exc:
        LOGGER.exception(
            "rustfs_object_ingestion_failed bucket=%s key=%s", request.bucket, request.key
        )
        raise _service_error(502, "RustFS object ingestion failed", exc) from exc


@app.post(
    "/v1/ingest/document",
    response_model=IngestionReportResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["ingestion"],
    summary="Chunk, embed, and ingest a directly supplied document",
)
async def ingest_document(request: DirectDocumentIngestRequest) -> IngestionReportResponse:
    try:
        return await get_service().ingest_document(request, (await ensure_ledger_schema()).ledger)
    except Exception as exc:
        LOGGER.exception("direct_document_ingestion_failed document_id=%s", request.document_id)
        raise _service_error(503, "Direct document ingestion failed", exc) from exc


@app.post(
    "/v1/rag",
    response_model=RagResponse,
    tags=["retrieval"],
    summary="Retrieve evidence and generate a grounded answer",
)
async def rag(request: RagRequest) -> RagResponse:
    try:
        return await get_service().rag(request.query, request.top_k, request.filters)
    except Exception as exc:
        LOGGER.exception("rag_request_failed")
        raise _service_error(503, "RAG service unavailable", exc) from exc


@app.post(
    "/v1/scoring",
    response_model=ScoringResponse,
    tags=["evaluation"],
    summary="Score an answer using the rag-bench Llama 3.1 methodology",
)
async def score(request: ScoringRequest) -> ScoringResponse:
    try:
        metrics = await get_service().score(
            request.query,
            request.contexts,
            request.answer,
            request.ground_truth,
            {metric.value for metric in request.metrics},
        )
        return ScoringResponse(metrics=metrics)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("scoring_request_failed")
        raise _service_error(503, "Evaluation service unavailable", exc) from exc


def _service_error(status_code: int, message: str, exc: Exception) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"message": message, "error_type": type(exc).__name__, "reason": str(exc)},
    )
