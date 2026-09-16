from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from dataclasses import asdict

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
from elite_rag.models import RawDocument
from elite_rag.runtime import Runtime
from elite_rag.rustfs import create_rustfs_client, iter_rustfs_documents, read_rustfs_document
from elite_rag.vector_store import CollectionReport, SearchFilter


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
