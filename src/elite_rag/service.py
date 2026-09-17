from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict
from uuid import uuid4

from elite_rag.api_models import (
    DirectDocumentIngestRequest,
    IngestionReportResponse,
    MetricsBreakdown,
    RagResponse,
    RetrievalFilters,
    RustFSObjectIngestRequest,
    RustFSPrefixIngestRequest,
    SourceContext,
)
from elite_rag.config import Settings
from elite_rag.evaluation import LlamaEvaluator
from elite_rag.ingestion import IngestionReport
from elite_rag.ingestion_ledger import IngestionLedger
from elite_rag.models import RawDocument
from elite_rag.runtime import Runtime
from elite_rag.rustfs import (
    create_rustfs_client,
    iter_rustfs_documents,
    iter_rustfs_objects,
    read_rustfs_document,
)
from elite_rag.vector_store import CollectionReport, SearchFilter

LOGGER = logging.getLogger(__name__)


class RAGService:
    """Application operations independent of HTTP transport and OpenAPI schemas."""

    def __init__(self, runtime: Runtime, settings: Settings, evaluator: LlamaEvaluator) -> None:
        self.runtime = runtime
        self.settings = settings
        self.evaluator = evaluator

    async def initialize_collection(self, recreate: bool) -> CollectionReport:
        return await asyncio.to_thread(self.runtime.store.ensure_collection, recreate)

    async def ingest_prefix(self, request: RustFSPrefixIngestRequest) -> IngestionReportResponse:
        report = await self.ingest_prefix_with_progress(request, None)
        return IngestionReportResponse(**asdict(report))

    async def ingest_prefix_with_progress(
        self,
        request: RustFSPrefixIngestRequest,
        progress: Callable[[IngestionReport], None] | None,
    ) -> IngestionReport:
        return await asyncio.to_thread(self._ingest_prefix_report, request, progress)

    async def ingest_durable_prefix_run(
        self, job_id: str, ledger: IngestionLedger
    ) -> IngestionReport:
        return await asyncio.to_thread(self._ingest_durable_prefix_run, job_id, ledger)

    async def ingest_object(self, request: RustFSObjectIngestRequest) -> IngestionReportResponse:
        return await asyncio.to_thread(self._ingest_object, request)

    async def ingest_document(
        self, request: DirectDocumentIngestRequest
    ) -> IngestionReportResponse:
        document = RawDocument(
            document_id=request.document_id,
            source=request.source,
            title=request.title or request.document_id,
            fields={"content": request.content},
            metadata=request.metadata,
        )
        return await asyncio.to_thread(self._ingest, [document], request.replace_existing)

    async def rag(
        self, query: str, top_k: int, filters: RetrievalFilters | None
    ) -> RagResponse:
        search_filter = _to_search_filter(filters)
        contexts = await self.runtime.retrieval.retrieve(query, search_filter, top_k)
        answer = await self.runtime.generation.generate(query, contexts)
        return RagResponse(
            answer=answer,
            source_contexts=[
                SourceContext(
                    document_id=context.document_id,
                    parent_id=context.parent_id,
                    text=context.text,
                    score=context.rerank_score
                    if context.rerank_score is not None
                    else context.vector_score,
                    metadata=context.metadata,
                )
                for context in contexts
            ],
        )

    async def score(
        self,
        query: str,
        contexts: list[str],
        answer: str,
        ground_truth: str | None,
        metrics: set[str],
    ) -> MetricsBreakdown:
        result = await asyncio.to_thread(
            self.evaluator.evaluate, query, answer, ground_truth, contexts, metrics
        )
        return MetricsBreakdown(**asdict(result))

    def _ingest_prefix(self, request: RustFSPrefixIngestRequest) -> IngestionReportResponse:
        report = self._ingest_prefix_report(request, None)
        return IngestionReportResponse(**asdict(report))

    def _ingest_prefix_report(
        self,
        request: RustFSPrefixIngestRequest,
        progress: Callable[[IngestionReport], None] | None,
    ) -> IngestionReport:
        client = create_rustfs_client(self.settings)
        documents = iter_rustfs_documents(
            client, request.bucket, request.prefix, request.source, request.limit
        )
        return self._ingest_report(documents, request.replace_existing, progress)

    def _ingest_durable_prefix_run(self, job_id: str, ledger: IngestionLedger) -> IngestionReport:
        run = ledger.get_run(job_id)
        if run is None:
            raise ValueError(f"Ingestion job '{job_id}' was not found in the ledger")
        client = create_rustfs_client(self.settings)
        self.runtime.store.ensure_collection()
        for object_info in iter_rustfs_objects(
            client, run.bucket, run.prefix, run.source, run.limit
        ):
            ledger.add_work_item(
                job_id,
                run.bucket,
                object_info.key,
                run.source,
                object_info.etag,
                object_info.size_bytes,
            )

        worker_id = uuid4()
        while True:
            item = ledger.claim_next_item(job_id, worker_id)
            if item is None:
                lease_delay = ledger.seconds_until_next_claim(job_id)
                if lease_delay is None:
                    break
                LOGGER.info(
                    "ingestion_work_item_lease_waiting job_id=%s wait_seconds=%.1f",
                    job_id,
                    lease_delay,
                )
                time.sleep(min(lease_delay, 5.0))
                continue
            LOGGER.info(
                (
                    "ingestion_work_item_claimed job_id=%s item_id=%s bucket=%s key=%s "
                    "attempt=%s"
                ),
                job_id,
                item.item_id,
                item.bucket,
                item.key,
                item.attempts,
            )
            try:
                document = read_rustfs_document(client, item.bucket, item.key, item.source)
                report = self.runtime.ingestion.ingest([document], run.replace_existing)
            except Exception as exc:
                ledger.retry_item(item, worker_id, exc)
                LOGGER.exception(
                    (
                        "ingestion_work_item_retryable_failure job_id=%s item_id=%s "
                        "key=%s error_type=%s"
                    ),
                    job_id,
                    item.item_id,
                    item.key,
                    type(exc).__name__,
                )
                raise
            permanent = report.documents_skipped_failed > 0
            ledger.complete_item(item, worker_id, report, permanent)
            LOGGER.info(
                (
                    "ingestion_work_item_completed job_id=%s item_id=%s key=%s status=%s "
                    "children_upserted=%s"
                ),
                job_id,
                item.item_id,
                item.key,
                "permanent_failure" if permanent else "completed",
                report.children_upserted,
            )
        completed = ledger.get_run(job_id)
        if completed is None:
            raise RuntimeError(f"Ingestion job '{job_id}' disappeared from the ledger")
        return completed.report

    def _ingest_object(self, request: RustFSObjectIngestRequest) -> IngestionReportResponse:
        client = create_rustfs_client(self.settings)
        document = read_rustfs_document(client, request.bucket, request.key, request.source)
        return self._ingest([document], request.replace_existing)

    def _ingest(
        self, documents: Iterable[RawDocument], replace_existing: bool
    ) -> IngestionReportResponse:
        report = self._ingest_report(documents, replace_existing, None)
        return IngestionReportResponse(**asdict(report))

    def _ingest_report(
        self,
        documents: Iterable[RawDocument],
        replace_existing: bool,
        progress: Callable[[IngestionReport], None] | None,
    ) -> IngestionReport:
        self.runtime.store.ensure_collection()
        return self.runtime.ingestion.ingest(documents, replace_existing, progress)


def _to_search_filter(filters: RetrievalFilters | None) -> SearchFilter | None:
    if filters is None or not (filters.sources or filters.projects):
        return None
    return SearchFilter(sources=tuple(filters.sources), projects=tuple(filters.projects))
