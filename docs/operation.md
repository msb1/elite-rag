# Operation

## Configuration

The application reads settings from explicit settings construction and the repository's
`.env` file; inherited shell environment variables and ambient secret files are not used by
`Settings`. Use `.env` for service endpoints and credentials. Important settings include:

| Area | Settings |
| --- | --- |
| Qdrant | `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION`, and the three vector names |
| RustFS | `RUSTFS_ENDPOINT` (or `S3_ENDPOINT_URL`), access and secret keys, bucket, prefix, and region |
| Sources and chunks | `SOURCE_TYPES`, `MULTI_SOURCE_TYPES`, `CHUNK_TARGET_TOKENS`, `CHUNK_MAX_TOKENS`, `CHUNK_OVERLAP_TOKENS`, and tokenizer settings |
| BGE embeddings | `EMBEDDING_BASE_URL` / `OPENAI_REMOTE_ENDPOINT`, model, dimension, and API key |
| Reranking | `BGE_RERANK_URL`, model, timeout, and maximum length |
| Answer model | `OPENAI_LOCAL_ENDPOINT`, `RAG_MODEL`, API key, temperature, and token limit |
| Evaluation judge | `EVAL_BASE_URL`, `EVAL_MODEL`, and `EVAL_API_KEY` |
| API and job persistence | `SERVER_HOST`, `SERVER_PORT`, `API_KEY`, and `INGESTION_DATABASE_URL` |

The default collection is `elite_rag_v2`, RustFS bucket is `enterprise-rag`, and object
prefix is `docs`. Tri-vector ingestion requires BGE-M3, 1024 dimensions, distinct vector
names, and `BM25_MODEL=Qdrant/bm25`. A collection with an incompatible schema is rejected;
choose a new collection or explicitly recreate it when appropriate.

## Start services

Start the separately deployed model service on its host, then start the API:

```bash
uv run --project remotes python remotes/remote-models.py
uv run elite-rag serve
```

The API defaults to `http://127.0.0.1:8080`. Its health endpoint is `GET /health`, and
interactive OpenAPI documentation is at `/docs`. When `API_KEY` is set, requests under
`/v1/` require `Authorization: Bearer <API_KEY>`; the health endpoint does not.

## Ingest documents

The standalone batch command reads configured source folders under the RustFS prefix and
creates or validates the tri-vector collection:

```bash
uv run python scripts/ingest_multi.py
uv run python scripts/ingest_multi.py --source fireflies
uv run python scripts/ingest_multi.py --limit 10
```

Use `--bucket`, `--prefix`, `--collection`, `--chunk-max-tokens`, or `--report` to override
the corresponding run values. The JSON report defaults to `results/ingestion_multi.json` and
is updated as objects are processed. Failed objects are reported while the command continues
to later objects.

The API also accepts direct document ingestion at `POST /v1/ingest/document`, an individual
RustFS object at `POST /v1/ingest/rustfs/object`, or a prefix job at
`POST /v1/ingest/rustfs/prefix`. Prefix ingestion returns a job ID; poll
`GET /v1/ingest/jobs/{job_id}` for status and counters. Set `INGESTION_DATABASE_URL` to use
PostgreSQL-backed job and object tracking that survives API restarts. Without it, the job
manager uses process memory.

`POST /v1/collections/initialize` creates or validates the configured collection. Its
`recreate` option deletes and recreates that collection, including its indexed passages.

## Retrieve and answer

Use the API's `POST /v1/retrieve` to inspect ranked passages without generation, or
`POST /v1/rag` to retrieve, rerank, and generate an answer. The command line supports a single
question:

```bash
uv run elite-rag rag "What does the indexed material say about ...?"
```

Retrieval embeds the original question once, fuses the three vector channels, and reranks
their candidates. Set `RETRIEVAL_NUMBER` for the candidate count and `RERANK_TOP_N` for the
number of passages passed to answer generation.

## Run benchmark and evaluation

The JSONL question set is configured by `QUESTIONS_FILE` and defaults to
`benchmark/questions.jsonl`. Benchmark output is appended to `DATASET_FILE` (default
`results/results.jsonl`) and processed question IDs are skipped on restart:

```bash
uv run elite-rag benchmark --limit 3
uv run elite-rag benchmark
```

Score generated rows and summarize them:

```bash
uv run elite-rag eval --limit 3
uv run elite-rag eval
uv run elite-rag analyze
```

Evaluation appends to `OUTPUT_FILE` (default `results/output.jsonl`) and skips records already
marked successful. Analysis writes metric summaries, a dashboard SVG, low-faithfulness cases,
and low-context-recall cases under `results/` by default. The corresponding API endpoints are
`POST /v1/benchmark`, `/v1/eval`, and `/v1/analyze`; `/v1/evaluate` scores one supplied
question, answer, context set, and gold answer.

## Troubleshooting signals

- A collection schema error means the configured Qdrant collection does not have the
  expected dense, learned sparse, and IDF BM25 vectors.
- An embedding error points to the configured BGE embeddings endpoint or an invalid response
  shape/vector dimension.
- An answer with a reranking warning means retrieval succeeded and generation can continue
  using the RRF ordering, but the reranking call failed.
- A benchmark or ingestion run can be resumed: completed question IDs are skipped in
  benchmark output, and complete passage revisions are skipped during ingestion.
