# Elite RAG

Elite RAG is an API-first parent-child enterprise retrieval system for the
[EnterpriseRAG-Bench](https://github.com/onyx-dot-app/EnterpriseRAG-Bench) corpus.
It uses local EmbeddingGemma embeddings, Qdrant, a local Jina reranker, Qwen generation,
and Llama 3.1 scoring through OpenAI-compatible services.

## Run the server

```bash
uv sync --extra dev
cp .env.example .env
uv run uvicorn elite_rag.api:app --host 0.0.0.0 --port 8080
```

Open [Swagger UI](http://localhost:8080/docs) to use and inspect every endpoint.
See [the API reference](docs/api.md) for request/response contracts and processing behavior.

## API operations

| Operation | Endpoint |
| --- | --- |
| Health check | `GET /health` |
| Initialize Qdrant collection | `POST /v1/collections/initialize` |
| Ingest a RustFS folder recursively | `POST /v1/ingest/rustfs/prefix` |
| Check a background prefix-ingestion job | `GET /v1/ingest/jobs/{job_id}` |
| Ingest one RustFS object | `POST /v1/ingest/rustfs/object` |
| Ingest supplied document text | `POST /v1/ingest/document` |
| Retrieve evidence and generate an answer | `POST /v1/rag` |
| Score an answer with Llama 3.1 | `POST /v1/scoring` |

Example direct-text ingestion:

```bash
curl -X POST http://localhost:8080/v1/ingest/document \
  -H 'content-type: application/json' \
  -d '{"document_id":"mcp-overview","source":"documentation","content":"Model Context Protocol is an open standard.","metadata":{"project":"platform"}}'
```

Example recursive RustFS ingestion:

```bash
curl -X POST http://localhost:8080/v1/ingest/rustfs/prefix \
  -H 'content-type: application/json' \
  -d '{"bucket":"enterprise-rag","prefix":"documents/jira","source":"jira"}'
```

Prefix ingestion returns `202` with a `job_id`; it does not hold the HTTP connection open. Poll
`GET /v1/ingest/jobs/{job_id}` for progress and errors. The append-only log is
`logs/elite-rag.log` by default and is controlled by `LOG_FILE` and `LOG_LEVEL`.

The configuration is a single gitignored `.env`; [.env.example](.env.example) documents
the Qdrant, RustFS, EmbeddingGemma, Jina, Qwen, and Llama settings.

## Verification

```bash
uv run ruff check src tests
uv run mypy src
uv run pytest -q
```
