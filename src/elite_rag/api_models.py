from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class CollectionInitializeRequest(BaseModel):
    recreate: bool = Field(
        default=False,
        description="Delete and recreate the collection. This permanently removes its points.",
    )


class CollectionResponse(BaseModel):
    collection_name: str
    created: bool
    recreated: bool
    indexed_fields: list[str]


class IngestionReportResponse(BaseModel):
    documents_seen: int
    documents_ingested: int
    documents_skipped_empty: int
    documents_skipped_failed: int
    parents_created: int
    children_upserted: int
    batches_upserted: int
    documents_replaced: int


class IngestionJobResponse(BaseModel):
    """State of a background RustFS-prefix ingestion job."""

    job_id: str
    status: str = Field(examples=["queued", "running", "completed", "failed"])
    bucket: str
    prefix: str
    submitted_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    report: IngestionReportResponse | None = None
    error: str | None = None


class RustFSPrefixIngestRequest(BaseModel):
    bucket: str = Field(min_length=1, examples=["enterprise-rag"])
    prefix: str = Field(min_length=1, examples=["documents/jira"])
    source: str | None = Field(default=None, examples=["jira"])
    limit: int | None = Field(default=None, ge=1)
    replace_existing: bool = False


class RustFSObjectIngestRequest(BaseModel):
    bucket: str = Field(min_length=1, examples=["enterprise-rag"])
    key: str = Field(min_length=1, examples=["documents/jira/PROJ-123.json"])
    source: str | None = Field(default=None, examples=["jira"])
    replace_existing: bool = False


class DirectDocumentIngestRequest(BaseModel):
    document_id: str = Field(min_length=1, examples=["doc_12345"])
    content: str = Field(min_length=1, examples=["Model Context Protocol is an open standard."])
    source: str = Field(default="manual", min_length=1, examples=["documentation"])
    title: str | None = Field(default=None, examples=["MCP overview"])
    metadata: dict[str, Any] = Field(default_factory=dict, examples=[{"project": "platform"}])
    replace_existing: bool = False


class RetrievalFilters(BaseModel):
    sources: list[str] = Field(default_factory=list, examples=[["jira", "linear"]])
    projects: list[str] = Field(default_factory=list, examples=[["hydra"]])

    @model_validator(mode="before")
    @classmethod
    def accept_native_payload_names(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        if "source" in normalized and "sources" not in normalized:
            source = normalized.pop("source")
            normalized["sources"] = source if isinstance(source, list) else [source]
        if "project_id" in normalized and "projects" not in normalized:
            project = normalized.pop("project_id")
            normalized["projects"] = project if isinstance(project, list) else [project]
        return normalized


class RagRequest(BaseModel):
    query: str = Field(min_length=1, max_length=16_000, examples=["What is MCP?"])
    top_k: int = Field(default=5, ge=1, le=50)
    filters: RetrievalFilters | None = None


class SourceContext(BaseModel):
    document_id: str
    parent_id: str
    text: str
    score: float | None
    metadata: dict[str, Any]


class RagResponse(BaseModel):
    answer: str
    source_contexts: list[SourceContext]


class MetricType(str, Enum):
    faithfulness = "faithfulness"
    answer_relevancy = "answer_relevancy"
    context_recall = "context_recall"
    context_precision = "context_precision"
    answer_correctness = "answer_correctness"


class ScoringRequest(BaseModel):
    query: str = Field(min_length=1, examples=["What is MCP?"])
    contexts: list[str] = Field(
        min_length=1, examples=[["Model Context Protocol is an open standard."]]
    )
    answer: str = Field(min_length=1, examples=["MCP stands for Model Context Protocol."])
    ground_truth: str | None = Field(
        default=None, examples=["Model Context Protocol is an open standard."]
    )
    metrics: list[MetricType] = Field(default_factory=lambda: list(MetricType))


class MetricsBreakdown(BaseModel):
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_recall: float | None = None
    context_precision: float | None = None
    answer_correctness: float | None = None


class ScoringResponse(BaseModel):
    metrics: MetricsBreakdown
