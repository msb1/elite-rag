from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from threading import Lock

from qdrant_client import QdrantClient, models

from elite_rag.config import Settings
from elite_rag.documents import Passage
from elite_rag.embedding import HybridVector


@dataclass(frozen=True, slots=True)
class CollectionReport:
    collection_name: str
    created: bool
    recreated: bool
    indexed_fields: tuple[str, ...]


class CollectionSchemaError(ValueError):
    pass


class PassageStore:
    INDEXED_FIELDS = ("document_id", "source", "project_id", "revision", "field_name", "section_id")

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            timeout=120,
        )
        self._collection_lock = Lock()

    def ensure_collection(self, recreate: bool = False) -> CollectionReport:
        with self._collection_lock:
            return self._ensure_collection(recreate)

    def _ensure_collection(self, recreate: bool) -> CollectionReport:
        name = self.settings.qdrant_collection
        existed = self.client.collection_exists(name)
        recreated = existed and recreate
        if existed and recreate:
            self.client.delete_collection(name)
            existed = False
        if not existed:
            self.client.create_collection(
                collection_name=name,
                vectors_config={
                    self.settings.qdrant_dense_vector_name: models.VectorParams(
                        size=self.settings.embedding_dimension,
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    self.settings.qdrant_sparse_vector_name: models.SparseVectorParams(
                        modifier=None
                    )
                },
            )
        else:
            self._validate_collection()
        for field in self.INDEXED_FIELDS:
            self.client.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )
        self.client.create_payload_index(
            collection_name=name,
            field_name="ingestion_ready",
            field_schema=models.PayloadSchemaType.BOOL,
            wait=True,
        )
        return CollectionReport(
            name, not existed, recreated, (*self.INDEXED_FIELDS, "ingestion_ready")
        )

    def _validate_collection(self) -> None:
        params = self.client.get_collection(self.settings.qdrant_collection).config.params
        vectors = params.vectors
        sparse = params.sparse_vectors or {}
        dense = (
            vectors.get(self.settings.qdrant_dense_vector_name)
            if isinstance(vectors, dict)
            else None
        )
        sparse_vector = sparse.get(self.settings.qdrant_sparse_vector_name)
        if (
            dense is None
            or dense.size != self.settings.embedding_dimension
            or dense.distance != models.Distance.COSINE
            or sparse_vector is None
            or sparse_vector.modifier is not None
        ):
            raise CollectionSchemaError(
                f"Collection {self.settings.qdrant_collection!r} does not match the configured "
                "BGE-M3 dense and sparse schema; use a new collection name"
            )

    def upsert(self, passages: Sequence[Passage], vectors: Sequence[HybridVector]) -> None:
        if len(passages) != len(vectors):
            raise ValueError("Each passage needs exactly one dense and sparse vector")
        points = []
        for passage, vector in zip(passages, vectors, strict=True):
            payload = {
                **passage.metadata,
                "document_id": passage.document_id,
                "passage_id": passage.point_id,
                "ordinal": passage.ordinal,
                "field_name": passage.field_name,
                "section_path": list(passage.section_path),
                "text": passage.text,
                "embedding_text": passage.embedding_text,
                "token_count": passage.token_count,
                "pipeline_version": self.settings.pipeline_signature,
                "ingestion_ready": False,
            }
            points.append(
                models.PointStruct(
                    id=passage.point_id,
                    vector={
                        self.settings.qdrant_dense_vector_name: vector.dense,
                        self.settings.qdrant_sparse_vector_name: models.SparseVector(
                            indices=vector.sparse_indices,
                            values=vector.sparse_values,
                        ),
                    },
                    payload=payload,
                )
            )
        if points:
            self.client.upsert(
                collection_name=self.settings.qdrant_collection, points=points, wait=True
            )

    def delete_other_revisions(self, document_id: str, revision: str) -> None:
        self.client.delete(
            collection_name=self.settings.qdrant_collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id", match=models.MatchValue(value=document_id)
                        )
                    ],
                    must_not=[
                        models.FieldCondition(
                            key="revision", match=models.MatchValue(value=revision)
                        )
                    ],
                )
            ),
            wait=True,
        )

    def delete_document(self, document_id: str) -> None:
        self.client.delete(
            collection_name=self.settings.qdrant_collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id", match=models.MatchValue(value=document_id)
                        )
                    ]
                )
            ),
            wait=True,
        )

    def has_revision(self, document_id: str, revision: str) -> bool:
        records, _ = self.client.scroll(
            collection_name=self.settings.qdrant_collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id", match=models.MatchValue(value=document_id)
                    ),
                    models.FieldCondition(key="revision", match=models.MatchValue(value=revision)),
                    models.FieldCondition(
                        key="ingestion_ready", match=models.MatchValue(value=True)
                    ),
                ]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return bool(records)

    def revision_count(self, document_id: str, revision: str, ready_only: bool = False) -> int:
        conditions = [
            models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id)),
            models.FieldCondition(key="revision", match=models.MatchValue(value=revision)),
        ]
        if ready_only:
            conditions.append(
                models.FieldCondition(key="ingestion_ready", match=models.MatchValue(value=True))
            )
        return self.client.count(
            collection_name=self.settings.qdrant_collection,
            count_filter=models.Filter(must=conditions),
            exact=True,
        ).count

    def commit_revision(self, document_id: str, revision: str, expected_count: int) -> None:
        written = self.revision_count(document_id, revision)
        if written != expected_count:
            raise RuntimeError(f"Incomplete revision for {document_id}: {written}/{expected_count}")
        selector = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id", match=models.MatchValue(value=document_id)
                ),
                models.FieldCondition(key="revision", match=models.MatchValue(value=revision)),
            ]
        )
        self.client.set_payload(
            collection_name=self.settings.qdrant_collection,
            payload={"ingestion_ready": True},
            points=selector,
            wait=True,
        )
        if self.revision_count(document_id, revision, ready_only=True) != expected_count:
            self.client.set_payload(
                collection_name=self.settings.qdrant_collection,
                payload={"ingestion_ready": False},
                points=selector,
                wait=True,
            )
            raise RuntimeError(f"Revision readiness verification failed for {document_id}")
