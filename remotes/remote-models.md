# Combined remote models service

`remote-models.py` is a single FastAPI process for the remote model computer. It leaves
`reranker.py` and `minicoil_service` untouched. It keeps the established reranker URL and adds
miniCOIL at the same host and port:

| Capability | Endpoint |
| --- | --- |
| Jina Reranker v3.5 | `POST /v1/rerank` |
| miniCOIL sparse embeddings | `POST /v1/embeddings/sparse` |
| Health and process RSS | `GET /health` |
| Swagger UI | `GET /docs` |

The reranker response remains `{ "results": [{ "index", "relevance_score" }] }`. Its `index`
is Jina's original position in the submitted `documents` array, which is the index Elite RAG uses
to associate a score with the correct parent context.

## Remote-computer setup

Copy these three files to one folder on the remote computer:

- `remote-models.py`
- `remote-models-requirements.txt`
- `remote-models.env.example` (rename it to `.env` if configuration overrides are needed)

Install into the Python environment that currently runs the Jina service:

```bash
python -m pip install -r remote-models-requirements.txt
```

Start it from that folder:

```bash
python remote-models.py
```

It binds to `0.0.0.0:8000`. Startup does not accept traffic until it has loaded both
`jinaai/jina-reranker-v3.5` on the configured device (default `cuda`) and
`Qdrant/minicoil-v1`. The first startup downloads model artifacts if they are not cached.

Set Elite RAG's single `.env` to:

```dotenv
JINA_RERANK_URL=http://REMOTE_HOST:8000/v1/rerank
SPARSE_EMBEDDING_URL=http://REMOTE_HOST:8000/v1/embeddings/sparse
```

The miniCOIL endpoint accepts no more than `MINICOIL_MAX_BATCH_SIZE` texts per request; Elite
RAG's default sparse batch size is 8, so it is compatible without a further change. The process
logs startup/shutdown, every rerank and embedding completion, failures, and resident memory.

## Smoke test

```bash
curl -sS http://localhost:8000/health
```

```bash
curl -sS -X POST http://localhost:8000/v1/embeddings/sparse \
  -H 'content-type: application/json' \
  -d '{"model":"Qdrant/minicoil-v1","mode":"query","texts":["ACME-404 support contact"]}'
```

```bash
curl -sS -X POST http://localhost:8000/v1/rerank \
  -H 'content-type: application/json' \
  -d '{"query":"What is the support contact?","documents":["Support code ACME-404.","A separate topic."],"top_n":2}'
```
