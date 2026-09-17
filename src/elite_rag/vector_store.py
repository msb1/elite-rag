from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import httpx
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from elite_rag.models import ChildChunk
from elite_rag.sparse_embedding import MiniCOILSparseEmbedder, SparseVector

LOGGER = logging.getLogger(__name__)


def _is_retryable_upsert_error(exc: Exception) -> bool:
    """Return whether re-sending deterministic Qdrant point IDs is safe and useful."""
    if isinstance(exc, (httpx.TransportError, ResponseHandlingException)):
        return True
    return (
        isinstance(exc, UnexpectedResponse)
        and exc.status_code is not None
        and 500 <= exc.status_code < 600
    )


@dataclass(frozen=True, slots=True)
class SearchFilter:
    sources: tuple[str, ...] = ()
    projects: tuple[str, ...] = ()

    @property
    def active(self) -> bool:
        return bool(self.sources or self.projects)


@dataclass(frozen=True, slots=True)
class CollectionReport:
    collection_name: str
    created: bool
    recreated: bool
    indexed_fields: tuple[str, ...]


class CollectionSchemaError(ValueError):
    """The existing collection cannot safely serve the configured retrieval pipeline."""


class QdrantVectorStore:
    """Hybrid Qdrant store with dense vectors and remote miniCOIL sparse vectors."""

    def __init__(
        self,
        url: str,
        collection_name: str,
        vector_size: int,
        api_key: str | None = None,
        client: Any | None = None,
        *,
        dense_vector_name: str = "dense",
        sparse_vector_name: str = "sparse",
        sparse_embedder: MiniCOILSparseEmbedder | None = None,
        hybrid_candidate_limit: int = 100,
        hybrid_enabled: bool = True,
        upsert_max_attempts: int = 5,
        upsert_initial_backoff_seconds: float = 0.5,
        upsert_max_backoff_seconds: float = 4.0,
    ) -> None:
        if upsert_max_attempts < 1:
            raise ValueError("upsert_max_attempts must be at least 1")
        if upsert_initial_backoff_seconds <= 0:
            raise ValueError("upsert_initial_backoff_seconds must be greater than 0")
        if upsert_max_backoff_seconds <= 0:
            raise ValueError("upsert_max_backoff_seconds must be greater than 0")
        if client is None:
            from qdrant_client import QdrantClient

            kwargs: dict[str, Any] = {"url": url}
            if api_key:
                kwargs["api_key"] = api_key
            client = QdrantClient(**kwargs)
        self.client = client
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.dense_vector_name = dense_vector_name
        self.sparse_vector_name = sparse_vector_name
        self.sparse_embedder = sparse_embedder
        self.hybrid_candidate_limit = hybrid_candidate_limit
        self.hybrid_enabled = hybrid_enabled
        self.upsert_max_attempts = upsert_max_attempts
        self.upsert_initial_backoff_seconds = upsert_initial_backoff_seconds
        self.upsert_max_backoff_seconds = upsert_max_backoff_seconds

    def ensure_collection(self, recreate: bool = False) -> CollectionReport:
        from qdrant_client.models import (
            Distance,
            Modifier,
            PayloadSchemaType,
            SparseVectorParams,
            VectorParams,
        )

        existed = self.client.collection_exists(self.collection_name)
        recreated = recreate and existed
        exists = existed
        if recreated:
            LOGGER.info("qdrant_collection_deleting collection=%s", self.collection_name)
            self.client.delete_collection(self.collection_name)
            exists = False
        if exists:
            self._validate_existing_collection()
        else:
            LOGGER.info(
                "qdrant_collection_creating collection=%s hybrid=%s",
                self.collection_name,
                self.hybrid_enabled,
            )
            vectors_config: Any
            sparse_vectors_config: Any = None
            if self.hybrid_enabled:
                vectors_config = {
                    self.dense_vector_name: VectorParams(
                        size=self.vector_size, distance=Distance.COSINE
                    )
                }
                sparse_vectors_config = {
                    self.sparse_vector_name: SparseVectorParams(modifier=Modifier.IDF)
                }
            else:
                vectors_config = VectorParams(size=self.vector_size, distance=Distance.COSINE)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_vectors_config,
            )
        indexed_fields = (
            "document_id",
            "parent_id",
            "metadata_filters.source",
            "metadata_filters.project_id",
            "metadata_filters.last_modified",
        )
        for field in indexed_fields:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
                wait=True,
            )
        LOGGER.info(
            "qdrant_collection_ready collection=%s created=%s recreated=%s hybrid=%s",
            self.collection_name,
            not existed or recreated,
            recreated,
            self.hybrid_enabled,
        )
        return CollectionReport(
            collection_name=self.collection_name,
            created=not existed or recreated,
            recreated=recreated,
            indexed_fields=indexed_fields,
        )

    def upsert(self, chunks: Sequence[ChildChunk], vectors: Sequence[Sequence[float]]) -> None:
        from qdrant_client.models import PointStruct
        from qdrant_client.models import SparseVector as QdrantSparseVector

        if len(chunks) != len(vectors):
            raise ValueError("Every child chunk must have exactly one dense vector")
        sparse_vectors = self._embed_sparse_documents(chunks)
        points = [
            PointStruct(
                id=chunk.child_id,
                vector=self._point_vectors(vector, sparse_vector, QdrantSparseVector),
                payload={
                    "document_id": chunk.document_id,
                    "parent_id": chunk.parent_id,
                    "child_content": chunk.text,
                    "vectorized_content": chunk.vector_text,
                    "parent_context": chunk.metadata["_parent_context"],
                    "metadata_filters": {
                        key: value
                        for key, value in chunk.metadata.items()
                        if not key.startswith("_")
                    },
                },
            )
            for chunk, vector, sparse_vector in zip(chunks, vectors, sparse_vectors, strict=True)
        ]
        if points:
            self._upsert_with_retries(points, chunks)
            for chunk in chunks:
                LOGGER.info(
                    (
                        "qdrant_point_upserted collection=%s point_id=%s document_id=%s "
                        "parent_id=%s dense_vector=%s sparse_vector=%s"
                    ),
                    self.collection_name,
                    chunk.child_id,
                    chunk.document_id,
                    chunk.parent_id,
                    self.dense_vector_name if self.hybrid_enabled else "default",
                    self.sparse_vector_name if self.hybrid_enabled else "disabled",
                )

    def _upsert_with_retries(self, points: list[Any], chunks: Sequence[ChildChunk]) -> None:
        document_ids = sorted({chunk.document_id for chunk in chunks})
        for attempt in range(1, self.upsert_max_attempts + 1):
            LOGGER.info(
                (
                    "qdrant_upsert_attempt collection=%s attempt=%s max_attempts=%s points=%s "
                    "document_ids=%s"
                ),
                self.collection_name,
                attempt,
                self.upsert_max_attempts,
                len(points),
                document_ids,
            )
            try:
                self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
            except Exception as exc:
                retryable = _is_retryable_upsert_error(exc)
                exhausted = attempt == self.upsert_max_attempts
                if not retryable or exhausted:
                    LOGGER.exception(
                        (
                            "qdrant_upsert_failed collection=%s attempt=%s max_attempts=%s "
                            "points=%s document_ids=%s retryable=%s exhausted=%s error_type=%s"
                        ),
                        self.collection_name,
                        attempt,
                        self.upsert_max_attempts,
                        len(points),
                        document_ids,
                        retryable,
                        exhausted,
                        type(exc).__name__,
                    )
                    raise
                delay = self._upsert_retry_delay(attempt)
                LOGGER.warning(
                    (
                        "qdrant_upsert_retrying collection=%s attempt=%s next_attempt=%s "
                        "max_attempts=%s points=%s document_ids=%s backoff_seconds=%.3f "
                        "error_type=%s error=%s"
                    ),
                    self.collection_name,
                    attempt,
                    attempt + 1,
                    self.upsert_max_attempts,
                    len(points),
                    document_ids,
                    delay,
                    type(exc).__name__,
                    exc,
                )
                time.sleep(delay)
                continue
            LOGGER.info(
                (
                    "qdrant_upsert_completed collection=%s attempt=%s points=%s "
                    "document_ids=%s"
                ),
                self.collection_name,
                attempt,
                len(points),
                document_ids,
            )
            return
        raise RuntimeError("Qdrant upsert retry loop completed without an outcome")

    def _upsert_retry_delay(self, failed_attempt: int) -> float:
        delay = self.upsert_initial_backoff_seconds * (2 ** (failed_attempt - 1))
        return float(min(delay, self.upsert_max_backoff_seconds))

    def delete_document(self, document_id: str) -> None:
        from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
                )
            ),
            wait=True,
        )

    def search(
        self,
        vector: Sequence[float],
        query_text: str,
        search_filter: SearchFilter | None,
        limit: int,
    ) -> list[Any]:
        if not self.hybrid_enabled:
            result = self.client.query_points(
                collection_name=self.collection_name,
                query=list(vector),
                query_filter=self._to_qdrant_filter(search_filter),
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            points = list(result.points)
            LOGGER.info(
                "qdrant_search_completed collection=%s mode=dense results=%s",
                self.collection_name,
                len(points),
            )
            return points

        from qdrant_client.models import Fusion, FusionQuery, Prefetch
        from qdrant_client.models import SparseVector as QdrantSparseVector

        candidate_limit = max(limit, self.hybrid_candidate_limit)
        query_filter = self._to_qdrant_filter(search_filter)
        sparse_query = self._embed_sparse_query(query_text)
        result = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(
                    query=list(vector),
                    using=self.dense_vector_name,
                    filter=query_filter,
                    limit=candidate_limit,
                ),
                Prefetch(
                    query=QdrantSparseVector(
                        indices=sparse_query.indices,
                        values=sparse_query.values,
                    ),
                    using=self.sparse_vector_name,
                    filter=query_filter,
                    limit=candidate_limit,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=candidate_limit,
            with_payload=True,
            with_vectors=False,
        )
        points = list(result.points)
        LOGGER.info(
            "qdrant_search_completed collection=%s mode=hybrid candidate_limit=%s results=%s",
            self.collection_name,
            candidate_limit,
            len(points),
        )
        return points

    def _validate_existing_collection(self) -> None:
        if not self.hybrid_enabled:
            return
        config = self.client.get_collection(self.collection_name).config.params
        dense_vectors = config.vectors
        sparse_vectors = config.sparse_vectors or {}
        dense_ready = isinstance(dense_vectors, dict) and self.dense_vector_name in dense_vectors
        sparse_ready = self.sparse_vector_name in sparse_vectors
        if not dense_ready or not sparse_ready:
            raise CollectionSchemaError(
                f"Collection '{self.collection_name}' is not hybrid-compatible. "
                "Create a new collection or call the initialize API with recreate=true."
            )

    def _point_vectors(
        self,
        vector: Sequence[float],
        sparse_vector: SparseVector | None,
        qdrant_sparse_vector: Any,
    ) -> Any:
        if not self.hybrid_enabled:
            return list(vector)
        if sparse_vector is None:
            raise RuntimeError("A sparse vector is required for hybrid indexing")
        return {
            self.dense_vector_name: list(vector),
            self.sparse_vector_name: qdrant_sparse_vector(
                indices=sparse_vector.indices,
                values=sparse_vector.values,
            ),
        }

    def _embed_sparse_documents(self, chunks: Sequence[ChildChunk]) -> list[SparseVector | None]:
        if not self.hybrid_enabled:
            return [None] * len(chunks)
        if self.sparse_embedder is None:
            raise RuntimeError("A remote miniCOIL sparse embedder is required for hybrid indexing")
        return [
            sparse_vector
            for sparse_vector in self.sparse_embedder.embed_documents(
                [chunk.vector_text for chunk in chunks]
            )
        ]

    def _embed_sparse_query(self, query_text: str) -> SparseVector:
        if self.sparse_embedder is None:
            raise RuntimeError("A remote miniCOIL sparse embedder is required for hybrid retrieval")
        return self.sparse_embedder.embed_query(query_text)

    @staticmethod
    def _to_qdrant_filter(search_filter: SearchFilter | None) -> Any:
        if search_filter is None or not search_filter.active:
            return None
        from qdrant_client.models import FieldCondition, Filter, MatchAny

        conditions: list[Any] = []
        if search_filter.sources:
            conditions.append(
                FieldCondition(
                    key="metadata_filters.source",
                    match=MatchAny(any=list(search_filter.sources)),
                )
            )
        if search_filter.projects:
            conditions.append(
                FieldCondition(
                    key="metadata_filters.project_id",
                    match=MatchAny(any=list(search_filter.projects)),
                )
            )
        return Filter(must=conditions)
