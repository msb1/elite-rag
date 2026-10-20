"""Qdrant tri-vector storage with BGE-M3 and server-side BM25 inference."""

from __future__ import annotations

from collections.abc import Sequence
from threading import Lock

from qdrant_client import QdrantClient, models

from elite_rag.config import Settings
from elite_rag.documents import Passage
from elite_rag.embedding import HybridVector
from elite_rag.store import CollectionReport, CollectionSchemaError, PassageStore

BM25_NAME = "sparse_bm25"
BM25_MODEL = "Qdrant/bm25"


def multi_settings(settings: Settings, collection: str | None = None) -> Settings:
    """Keep the source/chunk policy, and distinguish tri-vector revisions on resume."""
    if settings.embedding_model.lower() not in {"bge-m3", "baai/bge-m3"}:
        raise ValueError("Tri-vector ingestion and retrieval require BGE-M3")
    if settings.embedding_dimension != 1024:
        raise ValueError("BGE-M3 dense vectors must have 1024 dimensions")
    names = (
        settings.qdrant_dense_vector_name,
        settings.qdrant_sparse_vector_name,
        settings.qdrant_bm25_vector_name,
    )
    if any(not name.strip() for name in names) or len(set(names)) != 3:
        raise ValueError("Configure three distinct, non-empty Qdrant vector names in .env")
    if settings.bm25_model.lower() != "qdrant/bm25":
        raise ValueError("Server-side BM25 requires BM25_MODEL=Qdrant/bm25")
    if settings.multi_document_score not in {"max", "mean_top2"}:
        raise ValueError("MULTI_DOCUMENT_SCORE must be max or mean_top2")
    pipeline_version = settings.ingestion_pipeline_version
    suffix = "tri-vector-hierarchical-v1"
    if not pipeline_version.endswith(f":{suffix}"):
        pipeline_version = f"{pipeline_version}:{suffix}"
    return settings.model_copy(update={
        "qdrant_collection": collection or settings.qdrant_collection,
        "ingestion_pipeline_version": pipeline_version,
    })


def accepted_multi_pipeline_signatures(settings: Settings) -> set[str]:
    """Include the signature emitted by the older double-suffix ingest bug."""
    legacy_version = f"{settings.ingestion_pipeline_version}:tri-vector-hierarchical-v1"
    legacy = settings.model_copy(update={"ingestion_pipeline_version": legacy_version})
    return {settings.pipeline_signature, legacy.pipeline_signature}


def validate_multi_collection(client: QdrantClient, settings: Settings) -> None:
    """Reject incompatible collections rather than silently reusing dual-vector points."""
    params = client.get_collection(settings.qdrant_collection).config.params
    dense = (
        params.vectors.get(settings.qdrant_dense_vector_name)
        if isinstance(params.vectors, dict)
        else None
    )
    sparse = params.sparse_vectors or {}
    learned = sparse.get(settings.qdrant_sparse_vector_name)
    bm25 = sparse.get(settings.qdrant_bm25_vector_name)
    if (
        dense is None
        or dense.size != 1024
        or dense.distance != models.Distance.COSINE
        or learned is None
        or learned.modifier is not None
        or bm25 is None
        or bm25.modifier != models.Modifier.IDF
    ):
        raise CollectionSchemaError(
            f"Collection {settings.qdrant_collection!r} requires "
            f"{settings.qdrant_dense_vector_name} (1024, cosine), "
            f"{settings.qdrant_sparse_vector_name} (no modifier), and "
            f"{settings.qdrant_bm25_vector_name} (IDF); "
            "ingest into a new collection with ingest_multi.py"
        )


class MultiPassageStore(PassageStore):
    """Preserve payload indexes and the inherited revision commit/deletion protocol."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            timeout=120,
            # Forward Document inputs to the server instead of running FastEmbed locally.
            cloud_inference=True,
        )
        self._collection_lock = Lock()

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
                        index=models.SparseIndexParams(), modifier=None
                    ),
                    self.settings.qdrant_bm25_vector_name: models.SparseVectorParams(
                        index=models.SparseIndexParams(), modifier=models.Modifier.IDF
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
        validate_multi_collection(self.client, self.settings)

    def upsert(self, passages: Sequence[Passage], vectors: Sequence[HybridVector]) -> None:
        if len(passages) != len(vectors):
            raise ValueError("Each passage needs exactly one dense and sparse vector")
        points = []
        for passage, vector in zip(passages, vectors, strict=True):
            payload = {
                **passage.metadata,
                "document_id": passage.document_id,
                "parent_document_id": passage.document_id,
                "section_name": passage.section_path[-1] if passage.section_path else "",
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
                        # Give Qdrant the original chunk body for lexical matching.
                        self.settings.qdrant_bm25_vector_name: models.Document(
                            text=passage.text, model=self.settings.bm25_model
                        ),
                    },
                    payload=payload,
                )
            )
        if points:
            self.client.upsert(
                collection_name=self.settings.qdrant_collection, points=points, wait=True
            )

