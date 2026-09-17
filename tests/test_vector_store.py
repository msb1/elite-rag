from __future__ import annotations

import logging

import httpx
import pytest
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from elite_rag.models import ChildChunk
from elite_rag.sparse_embedding import SparseVector
from elite_rag.vector_store import QdrantVectorStore, SearchFilter


class FakeSparseEmbedder:
    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        return [SparseVector(indices=[1, 2], values=[0.5, 1.0]) for _ in texts]

    def embed_query(self, text: str) -> SparseVector:
        return SparseVector(indices=[1, 2], values=[0.5, 1.0])


def test_qdrant_payload_and_native_filters() -> None:
    store = QdrantVectorStore(
        "http://unused", "test", 4, client=QdrantClient(":memory:"), hybrid_enabled=False
    )
    with pytest.warns(UserWarning, match="Payload indexes have no effect"):
        report = store.ensure_collection()
    assert report.created
    assert not report.recreated
    assert "metadata_filters.source" in report.indexed_fields
    chunk = ChildChunk(
        child_id="00000000-0000-0000-0000-000000000001",
        parent_id="parent-1",
        document_id="doc-1",
        text="child",
        vector_text="embedded child",
        metadata={
            "source": "jira",
            "project_id": "hydra",
            "last_modified": "2026-09-01",
            "labels": ["incident", "security"],
            "_parent_context": "complete parent",
        },
    )
    store.upsert([chunk], [[1.0, 0.0, 0.0, 0.0]])

    hits = store.search(
        [1.0, 0.0, 0.0, 0.0],
        "embedded child",
        SearchFilter(sources=("jira",), projects=("hydra",)),
        5,
    )
    assert len(hits) == 1
    assert hits[0].payload["parent_context"] == "complete parent"
    assert hits[0].payload["metadata_filters"]["labels"] == ["incident", "security"]

    misses = store.search(
        [1.0, 0.0, 0.0, 0.0], "embedded child", SearchFilter(sources=("slack",)), 5
    )
    assert misses == []


class CapturingClient:
    def __init__(self) -> None:
        self.create_kwargs: dict[str, object] = {}
        self.upsert_points: list[models.PointStruct] = []
        self.query_kwargs: dict[str, object] = {}

    def collection_exists(self, collection_name: str) -> bool:
        return False

    def create_collection(self, **kwargs: object) -> None:
        self.create_kwargs = kwargs

    def create_payload_index(self, **kwargs: object) -> None:
        return None

    def upsert(self, **kwargs: object) -> None:
        self.upsert_points = list(kwargs["points"])  # type: ignore[arg-type]

    def query_points(self, **kwargs: object) -> object:
        self.query_kwargs = kwargs
        return type("Result", (), {"points": []})()


class FlakyUpsertClient:
    def __init__(self, failures: list[Exception]) -> None:
        self.failures = failures
        self.upsert_calls = 0
        self.point_id_batches: list[list[str]] = []

    def upsert(self, **kwargs: object) -> None:
        self.upsert_calls += 1
        points = kwargs["points"]
        self.point_id_batches.append([str(point.id) for point in points])  # type: ignore[union-attr]
        if self.failures:
            raise self.failures.pop(0)


def _chunk() -> ChildChunk:
    return ChildChunk(
        child_id="00000000-0000-0000-0000-000000000001",
        parent_id="parent-1",
        document_id="doc-1",
        text="child",
        vector_text="embedded child",
        metadata={"source": "jira", "_parent_context": "complete parent"},
    )


def test_hybrid_store_uses_named_dense_and_minicoil_sparse_vectors() -> None:
    client = CapturingClient()
    store = QdrantVectorStore(
        "http://unused", "hybrid", 4, client=client, sparse_embedder=FakeSparseEmbedder()  # type: ignore[arg-type]
    )
    store.ensure_collection()
    assert set(client.create_kwargs["vectors_config"]) == {"dense"}  # type: ignore[arg-type]
    sparse_config = client.create_kwargs["sparse_vectors_config"]  # type: ignore[assignment]
    assert sparse_config["sparse"].modifier == models.Modifier.IDF  # type: ignore[index,union-attr]

    chunk = ChildChunk(
        child_id="00000000-0000-0000-0000-000000000001",
        parent_id="parent-1",
        document_id="doc-1",
        text="child",
        vector_text="title: ticket | source: jira | ACME-404",
        metadata={"source": "jira", "_parent_context": "complete parent"},
    )
    store.upsert([chunk], [[1.0, 0.0, 0.0, 0.0]])
    point_vectors = client.upsert_points[0].vector
    assert point_vectors["dense"] == [1.0, 0.0, 0.0, 0.0]  # type: ignore[index]
    assert point_vectors["sparse"].indices == [1, 2]  # type: ignore[index,union-attr]
    assert point_vectors["sparse"].values == [0.5, 1.0]  # type: ignore[index,union-attr]

    store.search([1.0, 0.0, 0.0, 0.0], "What is ACME-404?", None, 25)
    prefetch = client.query_kwargs["prefetch"]
    assert len(prefetch) == 2  # type: ignore[arg-type]
    assert client.query_kwargs["query"].fusion == models.Fusion.RRF  # type: ignore[union-attr]
    assert client.query_kwargs["limit"] == 100
    assert prefetch[1].query.indices == [1, 2]  # type: ignore[index,union-attr]


def test_upsert_retries_transient_reset_with_identical_point_ids(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="elite_rag.vector_store")
    client = FlakyUpsertClient([httpx.ReadError("connection reset by peer")])
    store = QdrantVectorStore(
        "http://unused",
        "test",
        4,
        client=client,
        hybrid_enabled=False,
        upsert_max_attempts=5,
    )
    delays: list[float] = []
    monkeypatch.setattr("elite_rag.vector_store.time.sleep", delays.append)

    store.upsert([_chunk()], [[1.0, 0.0, 0.0, 0.0]])

    assert client.upsert_calls == 2
    assert client.point_id_batches == [
        ["00000000-0000-0000-0000-000000000001"],
        ["00000000-0000-0000-0000-000000000001"],
    ]
    assert delays == [0.5]
    assert "qdrant_upsert_retrying" in caplog.text
    assert "qdrant_upsert_completed" in caplog.text


def test_upsert_retries_qdrant_5xx_response(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FlakyUpsertClient(
        [UnexpectedResponse(503, "Service Unavailable", b"unavailable", httpx.Headers())]
    )
    store = QdrantVectorStore(
        "http://unused",
        "test",
        4,
        client=client,
        hybrid_enabled=False,
        upsert_max_attempts=2,
    )
    monkeypatch.setattr("elite_rag.vector_store.time.sleep", lambda _: None)

    store.upsert([_chunk()], [[1.0, 0.0, 0.0, 0.0]])

    assert client.upsert_calls == 2


def test_upsert_fails_after_transient_attempts_are_exhausted(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    client = FlakyUpsertClient(
        [httpx.ReadError("connection reset"), httpx.ReadError("connection reset")]
    )
    store = QdrantVectorStore(
        "http://unused",
        "test",
        4,
        client=client,
        hybrid_enabled=False,
        upsert_max_attempts=2,
    )
    delays: list[float] = []
    monkeypatch.setattr("elite_rag.vector_store.time.sleep", delays.append)

    with pytest.raises(httpx.ReadError, match="connection reset"):
        store.upsert([_chunk()], [[1.0, 0.0, 0.0, 0.0]])

    assert client.upsert_calls == 2
    assert delays == [0.5]
    assert "qdrant_upsert_failed" in caplog.text
    assert "exhausted=True" in caplog.text
