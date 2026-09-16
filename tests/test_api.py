from __future__ import annotations

from dataclasses import dataclass

from fastapi.testclient import TestClient

from elite_rag import api
from elite_rag.api_models import IngestionReportResponse, MetricsBreakdown, RagResponse


@dataclass
class FakeService:
    async def ingest_document(self, request: object) -> IngestionReportResponse:
        return IngestionReportResponse(
            documents_seen=1,
            documents_ingested=1,
            documents_skipped_empty=0,
            documents_skipped_failed=0,
            parents_created=1,
            children_upserted=2,
            batches_upserted=1,
            documents_replaced=0,
        )

    async def rag(self, query: str, top_k: int, filters: object) -> RagResponse:
        return RagResponse(answer=f"answer: {query}", source_contexts=[])

    async def score(
        self,
        query: str,
        contexts: list[str],
        answer: str,
        ground_truth: str | None,
        metrics: set[str],
    ) -> MetricsBreakdown:
        return MetricsBreakdown(faithfulness=0.9 if "faithfulness" in metrics else None)


def test_direct_document_ingestion_is_documented_and_returns_report(monkeypatch: object) -> None:
    monkeypatch.setattr(api, "get_service", lambda: FakeService())  # type: ignore[attr-defined]
    client = TestClient(api.app)
    response = client.post(
        "/v1/ingest/document",
        json={"document_id": "doc-1", "content": "A short document."},
    )
    assert response.status_code == 201
    assert response.json()["children_upserted"] == 2
    assert "/v1/ingest/rustfs/prefix" in client.get("/openapi.json").json()["paths"]


def test_rag_and_scoring_endpoints_return_structured_json(monkeypatch: object) -> None:
    monkeypatch.setattr(api, "get_service", lambda: FakeService())  # type: ignore[attr-defined]
    client = TestClient(api.app)
    rag_response = client.post("/v1/rag", json={"query": "What is MCP?", "top_k": 3})
    assert rag_response.status_code == 200
    assert rag_response.json()["answer"] == "answer: What is MCP?"

    score_response = client.post(
        "/v1/scoring",
        json={
            "query": "What is MCP?",
            "contexts": ["MCP is an open standard."],
            "answer": "MCP is an open standard.",
            "metrics": ["faithfulness"],
        },
    )
    assert score_response.status_code == 200
    assert score_response.json()["metrics"]["faithfulness"] == 0.9
