# Operations Guide

Elite RAG is operated exclusively through its FastAPI server. Start it with:

```bash
uv sync --extra dev
uv run uvicorn elite_rag.api:app --host 0.0.0.0 --port 8080
```

Interactive OpenAPI documentation is available at `http://localhost:8080/docs`.

## Collection lifecycle

Use `POST /v1/collections/initialize` with an empty JSON object to create or verify the
configured collection. Set `recreate` only for disposable collections; it deletes all
existing points before creating the collection again. After a successful create or recreation,
Elite RAG clears its PostgreSQL ingestion ledger so the ledger remains aligned with Qdrant.

## Ingestion

Use one of three endpoints:

| Endpoint | Use case |
| --- | --- |
| `POST /v1/ingest/rustfs/prefix` | Recursively ingest every document below a bucket prefix. |
| `GET /v1/ingest/jobs/{job_id}` | Monitor a submitted recursive-ingestion job. |
| `POST /v1/ingest/rustfs/object` | Ingest exactly one RustFS JSON or text object. |
| `POST /v1/ingest/document` | Ingest text supplied in the request body. |

Prefix ingestion returns `202 Accepted` immediately, then runs in the API process. Poll the job
endpoint until its state is `completed` or `failed`; its report contains documents processed,
parent contexts created, child vectors upserted, and replacements. The single-object and direct
document paths remain synchronous. Use `replace_existing` for a changed document with the same
stable ID.

The prefix job and its RustFS object checkpoints live in PostgreSQL, not FastAPI memory. Server
startup never starts or resumes ingestion; only an explicit prefix-ingestion API request can do so.
Each new inventory compares RustFS bucket, key, ETag,
collection, and pipeline version against the current successful-ingestion ledger; unchanged objects
are skipped before download and only new/changed objects receive work items. A work item held by a
stopped worker becomes reclaimable after `INGESTION_LEASE_SECONDS`. Document-specific
parsing/chunking failures become durable permanent work-item failures; infrastructure failures remain
retryable and fail the run after their current attempt.

Every RustFS object read and every successfully upserted Qdrant point is recorded in the
append-only `LOG_FILE` (default `logs/elite-rag.log`). `tail -f logs/elite-rag.log` is suitable
for progress monitoring. Failed requests and background jobs include complete exception traces in
that log; job status also exposes the exception type and message.

Qdrant upserts retry transport errors and HTTP 5xx responses with a bounded exponential backoff.
The log records every attempt, retry delay, point count, document IDs, and final outcome. Because
the same deterministic point IDs are reused, a retry is safe even if Qdrant accepted a write but
the client lost its response.

## Retrieval and scoring

`POST /v1/rag` returns a complete Qwen answer plus the retrieved parent contexts and
scores. It accepts optional `sources` and `projects` filters.

`POST /v1/scoring` runs the rag-bench Llama 3.1 evaluation methodology. It supports
faithfulness, answer relevancy, context recall, context precision, and answer correctness.
Metrics requiring a reference answer reject requests without `ground_truth`.

## Production considerations

The API process requires trusted-network access to Qdrant, RustFS, the LM Studio-compatible
EmbeddingGemma/Qwen/Llama services and the [combined remote models service](../services/remote-models.md)
on port `8000`. It exposes Jina at `/v1/rerank` and miniCOIL at `/v1/embeddings/sparse`. Put
authentication, authorization, rate limits, request-size limits, and TLS at the service boundary
before external exposure.
