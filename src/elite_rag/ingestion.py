from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from threading import Lock

from elite_rag.chunking import HierarchicalPassageChunker
from elite_rag.config import Settings
from elite_rag.documents import SourceDocument
from elite_rag.embedding import HybridEmbedder
from elite_rag.multi import MultiPassageStore, multi_settings
from elite_rag.store import CollectionReport
from elite_rag.tokenization import TokenCounter

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class IngestionReport:
    documents_seen: int = 0
    documents_ingested: int = 0
    documents_skipped_unchanged: int = 0
    documents_skipped_empty: int = 0
    documents_skipped_failed: int = 0
    parents_created: int = 0
    expected_passages: int = 0
    verified_passages: int = 0
    children_upserted: int = 0
    batches_upserted: int = 0
    documents_replaced: int = 0

    def add(self, other: IngestionReport) -> None:
        for key, value in asdict(other).items():
            setattr(self, key, getattr(self, key) + value)


class IngestionService:
    def __init__(self, settings: Settings) -> None:
        self.settings = multi_settings(settings)
        self.store = MultiPassageStore(self.settings)
        self.embedder = HybridEmbedder(
            self.settings.embedding_model,
            self.settings.embedding_dimension,
            self.settings.embedding_base_url,
            self.settings.embedding_api_key,
        )
        self._chunker: HierarchicalPassageChunker | None = None
        self._collection_report: CollectionReport | None = None
        self._collection_lock = Lock()

    def initialize_collection(self) -> CollectionReport:
        """Create or validate the collection once for this ingestion service."""
        with self._collection_lock:
            if self._collection_report is None:
                self._collection_report = self.store.ensure_collection()
            return self._collection_report

    @property
    def chunker(self) -> HierarchicalPassageChunker:
        if self._chunker is None:
            repository = self.settings.tokenizer_repository
            assert repository is not None
            self._chunker = HierarchicalPassageChunker(
                self.settings,
                TokenCounter(repository, self.settings.tokenizer_json_path).count,
            )
        return self._chunker

    def ingest(self, document: SourceDocument, replace_existing: bool = False) -> IngestionReport:
        report = IngestionReport(documents_seen=1)
        self.initialize_collection()
        passages = self.chunker.chunk(document)
        if not passages:
            self.store.delete_document(document.document_id)
            report.documents_skipped_empty = 1
            return report
        report.expected_passages = len(passages)
        revision = passages[0].metadata["revision"]
        if self.store.revision_count(document.document_id, revision, ready_only=True) == len(
            passages
        ):
            self.store.delete_other_revisions(document.document_id, revision)
            report.verified_passages = len(passages)
            report.documents_skipped_unchanged = 1
            return report
        try:
            for start in range(0, len(passages), self.settings.embedding_batch_size):
                batch = passages[start : start + self.settings.embedding_batch_size]
                vectors = self.embedder.embed_documents(
                    [passage.embedding_text for passage in batch]
                )
                self.store.upsert(batch, vectors)
                report.children_upserted += len(batch)
                report.batches_upserted += 1
            self.store.commit_revision(document.document_id, revision, len(passages))
            report.verified_passages = len(passages)
            self.store.delete_other_revisions(document.document_id, revision)
        except Exception:
            LOGGER.exception("ingestion_failed document_id=%s", document.document_id)
            raise
        report.documents_ingested = 1
        report.documents_replaced = int(replace_existing)
        LOGGER.info(
            "ingestion_completed document_id=%s revision=%s passages=%s",
            document.document_id,
            revision,
            len(passages),
        )
        return report
