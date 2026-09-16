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
existing points before creating the collection again.

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

Every RustFS object read and every successfully upserted Qdrant point is recorded in the
append-only `LOG_FILE` (default `logs/elite-rag.log`). `tail -f logs/elite-rag.log` is suitable
for progress monitoring. Failed requests and background jobs include complete exception traces in
that log; job status also exposes the exception type and message.

## Retrieval and scoring

`POST /v1/rag` returns a complete Qwen answer plus the retrieved parent contexts and
scores. It accepts optional `sources` and `projects` filters.

`POST /v1/scoring` runs the rag-bench Llama 3.1 evaluation methodology. It supports
faithfulness, answer relevancy, context recall, context precision, and answer correctness.
Metrics requiring a reference answer reject requests without `ground_truth`.

## Production considerations

The API process requires trusted-network access to Qdrant, RustFS, LM Studio-compatible
model services, and the Jina reranker. Put authentication, authorization, rate limits,
request-size limits, and TLS at the service boundary before external exposure.
