# Elite RAG HTTP API

Start the server:

```bash
uv run uvicorn elite_rag.api:app --host 0.0.0.0 --port 8080
```

Interactive Swagger documentation is served at `GET /docs`. ReDoc is served at `GET /redoc`,
and the machine-readable OpenAPI schema is served at `GET /openapi.json`.

All JSON requests require `Content-Type: application/json`. The API is not authenticated by
itself; put authentication and TLS in front of it before exposing it outside a trusted network.

The examples below use `http://localhost:8080`. Replace this with the API server's hostname or
IP address when calling it remotely.

## `GET /health`

Returns process liveness only. It does not contact Qdrant, RustFS, the embedding server, Jina,
Qwen, or Llama.

curl:

```bash
curl http://localhost:8080/health
```

Response `200`:

```json
{"status":"ok"}
```

## `POST /v1/collections/initialize`

Creates or verifies the configured `QDRANT_COLLECTION`. It also creates payload indexes for
`document_id`, `parent_id`, source, project, and last-modified metadata.

Request body:

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `recreate` | boolean | no, default `false` | If true, deletes the entire configured collection before rebuilding it. |

Safe request:

```json
{}
```

Destructive request:

```json
{"recreate":true}
```

curl, create or verify without deleting existing points:

```bash
curl -sS -X POST http://localhost:8080/v1/collections/initialize \
  -H 'Content-Type: application/json' \
  -d '{"recreate":false}'
```

curl, permanently delete and recreate the collection with the hybrid schema:

```bash
curl -sS -X POST http://localhost:8080/v1/collections/initialize \
  -H 'Content-Type: application/json' \
  -d '{"recreate":true}'
```

Processing steps:

1. Reads Qdrant settings from `.env`.
2. Checks whether the configured collection exists.
3. Optionally deletes it when `recreate` is true.
4. Creates a named dense cosine vector using `EMBEDDING_DIMENSION` and a named miniCOIL sparse
   vector with Qdrant's `IDF` modifier when absent.
5. Creates or verifies the native payload indexes.

Response `200`:

```json
{
  "collection_name":"elite_rag",
  "created":true,
  "recreated":false,
  "indexed_fields":[
    "document_id",
    "parent_id",
    "metadata_filters.source",
    "metadata_filters.project_id",
    "metadata_filters.last_modified"
  ]
}
```

`503` means the Qdrant operation failed or Qdrant was unreachable.
`409` means an existing dense-only collection must be explicitly recreated before it can serve
the hybrid dense/miniCOIL schema.

## `POST /v1/ingest/rustfs/prefix`

Recursively ingests every supported object under a RustFS bucket prefix. This is the endpoint
for a bucket and folder/key prefix. A prefix such as `documents/jira` includes all nested
children such as `documents/jira/2025/PROJ-1.json`.

Request body:

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `bucket` | string | yes | RustFS bucket name. |
| `prefix` | string | yes | Recursive RustFS object-key prefix. |
| `source` | string or null | no | Overrides the detected source for every object, for example `jira`. |
| `limit` | integer or null | no | Maximum number of objects to process; must be at least 1. |
| `replace_existing` | boolean | no, default `false` | Deletes each document's existing Qdrant points before replacing it. |

Request:

```json
{
  "bucket":"enterprise-rag",
  "prefix":"documents/jira",
  "source":"jira",
  "limit":100,
  "replace_existing":false
}
```

curl, recursively ingest a RustFS folder and every child folder:

```bash
curl -sS -X POST http://localhost:8080/v1/ingest/rustfs/prefix \
  -H 'Content-Type: application/json' \
  -d '{
    "bucket":"enterprise-rag",
    "prefix":"docs",
    "replace_existing":false
  }'
```

For a small diagnostic run, add `"limit": 10` to the JSON body. Do not include `source` unless
you intend to override the source detected from the RustFS key for every object.

Processing steps:

1. Connects to RustFS with `S3_ENDPOINT_URL`, `S3_ACCESS_KEY`, and `S3_SECRET_KEY`.
2. Paginates `list_objects_v2` for every object below the prefix.
3. Reads each JSON or text object from RustFS.
4. Converts JSON through the EnterpriseRAG-Bench field-declaration loader; text becomes a
   `RawDocument` using the first line as title.
5. Ensures the Qdrant collection and indexes exist.
6. Uses the source-aware parser, creates zero-overlap child chunks and bounded parent contexts,
   requests EmbeddingGemma dense vectors, asks FastEmbed to create miniCOIL sparse vectors, and
   upserts named dense/sparse child vectors into Qdrant.

Response `202 Accepted`:

```json
{
  "job_id":"7d0ec3fa-3d69-43f4-a0f7-3bed4ac5eb07",
  "status":"queued",
  "bucket":"enterprise-rag",
  "prefix":"documents/jira",
  "submitted_at":"2026-09-15T14:20:00Z",
  "started_at":null,
  "completed_at":null,
  "report":null,
  "error":null
}
```

The API runs this bulk operation in an in-process background job so long jobs cannot exhaust an
HTTP proxy timeout or look like a frozen request.

## `GET /v1/ingest/jobs/{job_id}`

Returns the live state, progress report, or failure details for a submitted RustFS-prefix job.
Replace the path parameter with the `job_id` returned by the prefix-ingestion POST.

curl:

```bash
curl -sS http://localhost:8080/v1/ingest/jobs/7d0ec3fa-3d69-43f4-a0f7-3bed4ac5eb07
```

Completed response example:

```json
{
  "job_id":"7d0ec3fa-3d69-43f4-a0f7-3bed4ac5eb07",
  "status":"completed",
  "bucket":"enterprise-rag",
  "prefix":"documents/jira",
  "submitted_at":"2026-09-15T14:20:00Z",
  "started_at":"2026-09-15T14:20:00Z",
  "completed_at":"2026-09-15T14:21:31Z",
  "report":{"documents_seen":100,"documents_ingested":99,"documents_skipped_empty":1,"documents_skipped_failed":0,"parents_created":125,"children_upserted":742,"batches_upserted":12,"documents_replaced":0},
  "error":null
}
```

A failed job has `status: "failed"` and an `error` with the exception type and reason. Job state
is in-memory, so submit a new job after an API-process restart. The full stack trace is retained
in `LOG_FILE`, whose default is `logs/elite-rag.log`.

The log records each `rustfs_object_read_started` / `rustfs_object_read_completed` event and each
`qdrant_point_upserted` event. Monitor it while ingestion runs:

```bash
tail -f logs/elite-rag.log
```

Documents that fail parsing or chunking are skipped instead of stopping a prefix job. Each is
recorded with its document ID, RustFS `object_key`, source, error type, error message, and UTC
timestamp in `MISSING_DOCS_FILE` (default `logs/missing_docs.json`). Infrastructure failures such
as an unavailable embedding server, RustFS, or Qdrant still fail the job because the application
cannot safely continue indexing while those dependencies are unavailable.

## `POST /v1/ingest/rustfs/object`

Ingests one RustFS object by its complete bucket and key.

Request body:

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `bucket` | string | yes | RustFS bucket name. |
| `key` | string | yes | Full RustFS object key, including filename. |
| `source` | string or null | no | Overrides source detection. |
| `replace_existing` | boolean | no, default `false` | Replaces points for the stable document ID. |

Request:

```json
{
  "bucket":"enterprise-rag",
  "key":"documents/jira/PROJ-123.json",
  "source":"jira",
  "replace_existing":true
}
```

curl:

```bash
curl -sS -X POST http://localhost:8080/v1/ingest/rustfs/object \
  -H 'Content-Type: application/json' \
  -d '{
    "bucket":"enterprise-rag",
    "key":"docs/confluence/engineering-overview.txt",
    "replace_existing":false
  }'
```

Processing is the same as prefix ingestion, except it calls `get_object` once rather than
listing a prefix. The response is the same `IngestionReportResponse` schema and normally has
`documents_seen: 1`.

## `POST /v1/ingest/document`

Ingests document text supplied directly in the HTTP request. Use this for application-originated
documents that do not already reside in RustFS.

Request body:

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `document_id` | string | yes | Stable document identifier. Reuse it with `replace_existing` for updates. |
| `content` | string | yes | Raw document text to parse, chunk, embed, and store. |
| `source` | string | no, default `manual` | Source type. Known source values select specialized parsers; unknown values use field parsing. |
| `title` | string or null | no | Document title. Defaults to `document_id`. |
| `metadata` | object | no, default `{}` | Arbitrary JSON metadata. `project`, `project_id`, and date-like fields participate in normalized filters. |
| `replace_existing` | boolean | no, default `false` | Deletes existing points for this document ID before upserting. |

Request:

```json
{
  "document_id":"mcp-overview",
  "source":"documentation",
  "title":"MCP overview",
  "content":"Model Context Protocol is an open standard for connecting AI applications to tools and context.",
  "metadata":{"project":"platform","last_modified":"2026-09-15"},
  "replace_existing":true
}
```

curl:

```bash
curl -sS -X POST http://localhost:8080/v1/ingest/document \
  -H 'Content-Type: application/json' \
  -d '{
    "document_id":"mcp-overview",
    "source":"documentation",
    "title":"MCP overview",
    "content":"Model Context Protocol is an open standard for connecting AI applications to tools and context.",
    "metadata":{"project":"platform","last_modified":"2026-09-16"},
    "replace_existing":true
  }'
```

Processing steps:

1. Creates an in-memory `RawDocument` from the submitted values.
2. Ensures the configured Qdrant collection exists.
3. Parses content, normalizes metadata, chunks parents and children, embeds child inputs, and
   upserts them to Qdrant.

The response is the same `IngestionReportResponse` schema described above. `503` means a Qdrant
or model service was unavailable.

## `POST /v1/rag`

Retrieves parent contexts from Qdrant, reranks them using local Jina, and generates a grounded
answer using Qwen.

Request body:

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `query` | string | yes | User question; 1 to 16,000 characters. |
| `top_k` | integer | no, default `5` | Number of final parent contexts; 1 through 50. |
| `filters` | object or null | no | Optional hard filters for source and project. |
| `filters.sources` | string array | no | Source values such as `jira`, `linear`, or `slack`. |
| `filters.projects` | string array | no | Project identifiers such as `hydra`. |

`filters.source` and `filters.project_id` are also accepted as singular shorthand fields.

Request:

```json
{
  "query":"What changed in the authentication proxy?",
  "top_k":5,
  "filters":{"sources":["jira","github"],"projects":["hydra"]}
}
```

curl:

```bash
curl -sS -X POST http://localhost:8080/v1/rag \
  -H 'Content-Type: application/json' \
  -d '{
    "query":"What changed in the authentication proxy?",
    "top_k":5,
    "filters":{"sources":["jira","github"],"projects":["hydra"]}
  }'
```

Processing steps:

1. Extracts any deterministic natural-language filters from the query.
2. Applies explicit request filters as hard source/project constraints.
3. Embeds the query with local EmbeddingGemma and FastEmbed miniCOIL.
4. Runs Qdrant dense and sparse prefetches, fuses the configured candidate rankings natively with
   RRF, and collapses matching child vectors into unique parent contexts.
5. Calls the local Jina reranker at `JINA_RERANK_URL`.
6. If reranker confidence is below the configured threshold, executes exactly one corrective
   retrieval pass using the deterministic fallback query.
7. Builds the injection-resistant evidence prompt and calls Qwen `RAG_MODEL`.

Response `200`:

```json
{
  "answer":"The authentication proxy change was ...",
  "source_contexts":[
    {
      "document_id":"dsid_abc123",
      "parent_id":"c4e5...",
      "text":"DOCUMENT ID: ...",
      "score":0.91,
      "metadata":{"source":"jira","project_id":"hydra","last_modified":"2026-09-15"}
    }
  ]
}
```

`score` is the Jina relevance score when reranking succeeded; otherwise it is the Qdrant vector
similarity score. `503` means a retrieval, reranking, Qdrant, or Qwen service failed.

## `POST /v1/scoring`

Scores a supplied answer with the rag-bench Llama 3.1 evaluator. It does not perform retrieval or
generation; the caller supplies the question, contexts, answer, and optional reference answer.

Request body:

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `query` | string | yes | Original user question. |
| `contexts` | string array | yes | Retrieved context texts used to answer the query. At least one is required. |
| `answer` | string | yes | Answer to evaluate. |
| `ground_truth` | string or null | conditional | Required for context recall, context precision, and answer correctness. |
| `metrics` | enum array | no | Metrics to run. Defaults to all five metrics. |

Metric values:

| Value | Method |
| --- | --- |
| `faithfulness` | Extract answer claims, then have Llama verify each claim against the contexts. |
| `answer_relevancy` | Have Llama reconstruct the implied question, then score semantic match to the original query. |
| `context_recall` | Extract ground-truth facts, then verify each fact appears in the contexts. |
| `context_precision` | Score each context sentence for usefulness against the question and ground truth. |
| `answer_correctness` | Ask Llama for TP/FP/FN JSON and calculate `TP / (TP + 0.5 * (FP + FN))`. |

Faithfulness-only request:

```json
{
  "query":"What is MCP?",
  "contexts":["Model Context Protocol is an open standard."],
  "answer":"MCP is an open standard.",
  "metrics":["faithfulness"]
}
```

curl, faithfulness-only scoring:

```bash
curl -sS -X POST http://localhost:8080/v1/scoring \
  -H 'Content-Type: application/json' \
  -d '{
    "query":"What is MCP?",
    "contexts":["Model Context Protocol is an open standard."],
    "answer":"MCP is an open standard.",
    "metrics":["faithfulness"]
  }'
```

Full evaluation request:

```json
{
  "query":"What is MCP?",
  "contexts":["Model Context Protocol is an open standard for connecting AI systems to tools."],
  "answer":"MCP is an open standard for connecting AI systems to tools.",
  "ground_truth":"Model Context Protocol is an open standard for connecting AI systems to tools.",
  "metrics":[
    "faithfulness",
    "answer_relevancy",
    "context_recall",
    "context_precision",
    "answer_correctness"
  ]
}
```

curl, full evaluation:

```bash
curl -sS -X POST http://localhost:8080/v1/scoring \
  -H 'Content-Type: application/json' \
  -d '{
    "query":"What is MCP?",
    "contexts":["Model Context Protocol is an open standard for connecting AI systems to tools."],
    "answer":"MCP is an open standard for connecting AI systems to tools.",
    "ground_truth":"Model Context Protocol is an open standard for connecting AI systems to tools.",
    "metrics":["faithfulness","answer_relevancy","context_recall","context_precision","answer_correctness"]
  }'
```

Response `200`:

```json
{
  "metrics":{
    "faithfulness":1.0,
    "answer_relevancy":0.95,
    "context_recall":1.0,
    "context_precision":0.97,
    "answer_correctness":1.0
  }
}
```

Metrics not requested are returned as `null`. `422` means a selected metric requires missing
`ground_truth`. `503` means the Llama evaluator service was unavailable.
