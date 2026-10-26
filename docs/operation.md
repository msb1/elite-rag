# Operation

Run commands from the repository root so `.env` and relative input/output paths resolve
consistently. The [README](../README.md) gives the end-to-end sequence;
[ragas.md](../ragas.md) is the exact scoring reference.

## Configuration

Install the application and test tools with Python 3.10+ and `uv sync --extra dev`.
`Settings` reads explicit constructor values first, then `.env`. It ignores inherited shell
environment variables and ambient secret files. Edit `.env` rather than using shell variable
assignments to configure a run. The following values are **code defaults**; a checkout's
`.env` may override them.

| Area | Settings and behavior |
| --- | --- |
| Qdrant | `QDRANT_URL` (localhost:6333), optional `QDRANT_API_KEY`, `QDRANT_COLLECTION` (`elite_rag_v2`) |
| Vector names | `QDRANT_DENSE_VECTOR_NAME=dense`, `QDRANT_SPARSE_VECTOR_NAME=sparse`, `QDRANT_BM25_VECTOR_NAME=sparse_bm25`, `BM25_MODEL=Qdrant/bm25` |
| RustFS/S3 | `S3_ENDPOINT_URL` / `RUSTFS_ENDPOINT` (localhost:9000), `S3_ACCESS_KEY` / `RUSTFS_ACCESS_KEY`, `S3_SECRET_KEY` / `RUSTFS_SECRET_KEY`, `S3_REGION` / `RUSTFS_REGION` (`us-east-1`) |
| Source location | `S3_BUCKET` / `RUSTFS_BUCKET` (`enterprise-rag`), `S3_PATH_MARKDOWN_PREFIX` / `RUSTFS_PREFIX` (`docs`) |
| Source selection | `SOURCE_TYPES` recognizes object path segments; `MULTI_SOURCE_TYPES` selects batch folders and must be a subset |
| BGE embeddings | `OPENAI_REMOTE_ENDPOINT` / `EMBEDDING_BASE_URL` (`http://192.168.1.50:8000/v1`), `EMBEDDING_MODEL=bge-m3`, `EMBEDDING_DIMENSION=1024`, `EMBEDDING_API_KEY`, `EMBEDDING_BATCH_SIZE=16` |
| Tokenizer and chunks | `TOKENIZER_REPOSITORY` (inferred as `BAAI/bge-m3`), optional `TOKENIZER_JSON_PATH`, `MULTI_CHUNK_MAX_TOKENS=300`, `CHUNK_OVERLAP_TOKENS=40`, `BGE_EMBEDDING_MAX_LENGTH=1024` |
| Generic policy | `CHUNK_TARGET_TOKENS=320`, `CHUNK_MAX_TOKENS=420`, `INGESTION_PIPELINE_VERSION=v3`; generic chunk settings affect signature/validation, while active hierarchical body packing uses the multi limit |
| Retrieval | `RETRIEVAL_NUMBER=50`, `RERANK_TOP_N=5` unique parent documents after child reranking/deduplication; `MULTI_DOCUMENT_SCORE` is validated but not used |
| Reranker client | `BGE_RERANK_URL` (`http://192.168.1.50:8000/v1/rerank`), `RERANK_TIMEOUT_SECONDS=180`; client sends query/documents/top_n, not model/max_length |
| Answer model | `OPENAI_LOCAL_ENDPOINT` (`http://127.0.0.1:1234/v1`), `RAG_MODEL=qwen2.5-7b-instruct-mlx`, `LLM_API_KEY`, `RAG_TEMPERATURE=0.7`, `RAG_MAX_TOKENS=2048` |
| Judge | `EVAL_BASE_URL` / `OPENAI_LOCAL_ENDPOINT`, `EVAL_MODEL=meta-llama-3.1-8b-instruct`, `EVAL_API_KEY`; judge temperature is fixed at 0 |
| HTTP and jobs | `SERVER_HOST=127.0.0.1`, `SERVER_PORT=8080`, optional `API_KEY`, optional PostgreSQL `INGESTION_DATABASE_URL` |
| Logging | `LOG_FILE=logs/elite-rag.log`, `LOG_LEVEL=INFO`; configured during API startup |

For slash-separated aliases, the first name takes precedence when both are set. Default
sources are `confluence,fireflies,github,gmail,google_drive,hubspot,jira,linear,slack`.
Tri-vector paths require BGE-M3, 1024 dimensions, three distinct non-empty vector names, and
Qdrant BM25. An existing incompatible collection fails validation; ingest into a new
collection rather than assuming automatic schema migration.

Output configuration:

| Setting | Code default |
| --- | --- |
| `QUESTIONS_FILE` | `benchmark/questions.jsonl` |
| `MULTI_INGESTION_REPORT` | `results/ingestion_multi.json` |
| `MULTI_RETRIEVAL_OUTPUT` | `results/multi_retrieval_results.jsonl` |
| `DATASET_FILE` | `results/results.jsonl` |
| `PROGRESS_FILE` | `results/progress.json` |
| `OUTPUT_FILE` | `results/output.jsonl` |
| `ANALYSIS_OUTPUT_FILE` | `results/analysis.json` |
| `DASHBOARD_OUTPUT_FILE` | `results/rag_metrics_dashboard.svg` |
| `HALLUCINATIONS_OUTPUT_FILE` | `results/hallucinations.jsonl` |
| `RETRIEVAL_MISSES_OUTPUT_FILE` | `results/retrieval_misses.jsonl` |

## Start services

Provide RustFS/S3, Qdrant with server-side BM25 inference, BGE embeddings/reranker, an answer
endpoint, and a judge endpoint. They can be separate hosts. The repo does not launch those
storage and language-model services as part of the CLI.

The bundled BGE service uses its own `remotes/pyproject.toml` (Python 3.11–3.13). On its host,
from the repository root:

```bash
uv run --project remotes python remotes/remote-models.py
```

It reads the working directory's `.env`, binds `0.0.0.0` on `PORT` (default 8000), and exposes
`/v1/embeddings`, `/v1/rerank`, `/v1/models`, and `/health`. `RERANK_MODEL`,
`RERANK_MAX_LENGTH`, `RERANK_BATCH_SIZE`, and `RERANKER_DEVICE` configure that service;
the reranker device defaults to `cuda`. `BGE_MODEL_NAME`, `BGE_EMBEDDING_DEVICE`, and
`BGE_EMBEDDING_MAX_LENGTH` configure its embedder. Keep client/server length limits aligned.
The embedding response must include both dense and learned sparse vectors, even for evaluation.

Start the optional application API:

```bash
uv run elite-rag serve
```

API startup initializes/validates Qdrant and resumes persisted ingestion jobs. `/health`
returns process status, not a dependency probe. The direct batch scripts do not need this API.

## Ingest documents

```bash
uv run python scripts/ingest_multi.py --limit 10
uv run python scripts/ingest_multi.py
uv run python scripts/ingest_multi.py --source fireflies
```

Objects are listed under `<prefix>/<source>/` for each selected source, accepting `.json`,
`.txt`, and `.md`. Keys must contain exactly one distinct configured source segment.
`--limit` counts listed objects across selected sources, including unchanged and failed ones.
Flags also include `--bucket`, `--prefix`, `--collection`, `--chunk-max-tokens`, and `--report`.
If using a collection override, configure retrieval to use the same collection afterward.

The script updates its JSON report after each object and overwrites that report on a new run.
It continues after individual failures, but exits 1 for failures, no objects, or a run-level
error. A limited successful run does not mean the entire source corpus has been indexed.

Rerunning lists from the beginning; complete ready revisions are skipped. Revision writes
use deterministic passage IDs and publish only after verifying counts. Empty parsed documents
delete points for that document. The script does not use the API's PostgreSQL job tracker.

## Test retrieval

```bash
uv run python scripts/multi_retrieval.py --input benchmark/questions.jsonl --limit 3 --output results/retrieval_check.jsonl
uv run python scripts/multi_retrieval.py
```

This requires Qdrant, BGE embeddings, and the reranker. It does not call the answer model
or judge. Output includes `qdrant_candidate_count`, `reranker_candidate_count`,
`qdrant_documents`, `reranked_documents`, `expected_document_ranks`,
`reranked_expected_document_ranks`, parent-document hit@5 fields, and `retrieval_status`.
Document lists deduplicate by
parent document ID in child-hit order, so their ranks are parent-document ranks. Reranking keeps all candidates here,
without `RERANK_TOP_N` truncation. A reranker failure marks the test row as an error; unlike
normal RAG, this script does not fall back to RRF. It exits 1 if any row fails.

Every run overwrites output and its sibling `.manifest.json`, which records input hash,
collection, retrieval count, vector names, fusion, reranker model setting, and run status.
It does not resume. Missing expected-document ranks mean those IDs were absent from that
stage's reported candidates, not a zero-based rank.

## Generate benchmark answers

```bash
uv run elite-rag rag "What does the indexed material say about ...?"
uv run python scripts/rag.py --benchmark --limit 3 --output results/questions_run.jsonl
uv run python scripts/rag.py --benchmark --output results/questions_run.jsonl
```

Set `QUESTIONS_FILE` in `.env` to change the benchmark input. `--question-id` selects one
question, but still skips it if already saved. `--limit` counts newly generated records.
`elite-rag benchmark` supports the same limit/question-id/output flags.

Generation appends benchmark metadata, answer, contexts, sources, warnings, and
`ground_truth=gold_answer`. It uses saved question IDs to resume; `PROGRESS_FILE` is only a
marker. An exception stops the run, with earlier flushed rows retained. Use a new output for
changed experiments: resume does not check input content, model, or collection changes.
Use a new output filename to regenerate answers after changing retrieval/context behavior;
existing question IDs are skipped and old saved contexts are not rewritten.
No query rewriting is performed. Reranked child hits are expanded to their parent documents,
deduplicated by parent document ID, then limited to `RERANK_TOP_N` unique parents before
generation. Reranker failure preserves RRF order and adds a warning. No valid parent contexts
returns the fixed abstention without calling the answer endpoint.

## Evaluate answers

```bash
uv run python scripts/eval.py --input results/questions_run.jsonl --output results/evaluation_current.jsonl --limit 3
uv run python scripts/eval.py --input results/questions_run.jsonl --output results/evaluation_current.jsonl
```

Evaluation requires only the saved answers, judge, and hybrid embedding service. It reuses
RAGAS 0.4.3 prompts/schemas and calculates scores locally. See [ragas.md](../ragas.md) for
formulas, validation, and edge cases. `elite-rag eval` accepts the same flags.

The implementation ID is `ragas-0.4.3-prompts-json-parent-context-v3`. The runner rejects an output
containing any successful row with another/missing ID; use a new output for older results.
It keeps the latest row per question ID, skips successful rows, and retries errors, replacing
their rows via temporary-file rewrite after every attempt. `--limit` counts attempts.
Existing output rows absent from the new input are retained.

Malformed prompt output receives one correction retry. Examine `evaluation_status`,
`evaluation_error`, `evaluation_stage`, `evaluation_raw_responses`, and
`evaluation_prompt_retries`. A successful row can include retry diagnostics. The runner
returns 0 even if some rows failed; inspect statuses before interpreting summary means.
Changing the judge or answer content does not invalidate already successful IDs automatically.

## Final summary analysis

```bash
uv run python scripts/analyze.py --input results/evaluation_current.jsonl
```

`elite-rag analyze --input ...` is equivalent. Analysis is local and needs no model/storage
services. It deduplicates by question ID, retaining the latest row, then writes:

- `ANALYSIS_OUTPUT_FILE`: successful-record metric summaries and document coverage summaries;
- `DASHBOARD_OUTPUT_FILE`: SVG mean-score bars;
- `HALLUCINATIONS_OUTPUT_FILE`: successful rows with faithfulness < 0.30;
- `RETRIEVAL_MISSES_OUTPUT_FILE`: successful rows with context recall < 0.30;
- `<analysis stem>_document_retrieval.jsonl` beside the summary: per-question document details.

`--input` changes only the input. It does **not** select a new output directory or prevent
replacement of the configured analysis artifacts. Change output settings in `.env` to retain
separate reports.

Read `input_records` and `records` to assess judge completion. Metric summaries report count,
mean, minimum, maximum, and sample standard deviation. Document summaries compare expected
IDs with final saved parent source IDs even for judge errors, report missing-label/ID exclusions,
and break down by question and source type. Cutoffs at 1/3/5 count up to k unique parent-document
ranks; short and empty results remain in the denominator as misses.
Use `at_k["5"].target_doc_hit_rate` (console label `target_doc_hit_rate@5`) for the success
criterion that at least one expected parent ID was retrieved in the top five parent ranks;
document recall measures the fraction of all expected parent IDs found. See
[ragas.md](../ragas.md#deterministic-document-retrieval-metrics).

Low judge context recall is a review signal rather than a confirmed parent-document retrieval miss.
Compare final source coverage with candidate/reranked document lists from the separate
retrieval test to investigate losses. That test is a separate run, not a saved trace of the
benchmark request. For `info_not_found`, interpret abstention separately. Analysis accepts
historical files without checking implementation IDs, so keep incompatible score sets apart.
The final output is a numeric summary and review exports, not an LLM-written assessment.

## HTTP API

The default base is `http://127.0.0.1:8080`, with schemas at `/docs`. Non-empty `API_KEY`
protects `/v1/` routes using `Authorization: Bearer <API_KEY>`; `/health` and docs are outside
that check.

| Method / path | Behavior |
| --- | --- |
| `GET /health` | Process status |
| `POST /v1/collections/initialize` | Create/validate collection; `{ "recreate": true }` deletes and recreates indexed data |
| `POST /v1/ingest/document` | Synchronous supplied document ingestion |
| `POST /v1/ingest/rustfs/object` | Synchronous one-object ingestion |
| `POST /v1/ingest/rustfs/prefix` | Return 202 with a background job ID |
| `GET /v1/ingest/jobs/{job_id}` | Job status and counters |
| `POST /v1/retrieve` | Reranked child hits expanded and deduplicated to parent contexts; no answer generation |
| `POST /v1/rag` | Grounded answer, contexts, sources, warnings |
| `POST /v1/evaluate` | Score supplied question/answer/contexts/ground_truth; no batch persistence |
| `POST /v1/benchmark` | Run generation with configured paths and optional `{ "limit": 3 }` |
| `POST /v1/eval` | Run evaluation with configured paths and optional limit |
| `POST /v1/analyze` | Generate analysis from configured evaluation file |

For an API with no configured key:

```bash
curl -sS http://127.0.0.1:8080/health
curl -sS http://127.0.0.1:8080/v1/retrieve -H 'Content-Type: application/json' -d '{"question":"What does the indexed material say about ...?"}'
uv run python scripts/rag_client.py "What does the indexed material say about ...?"
uv run python scripts/evaluate_client.py /path/to/answer_record.json
```

The Python clients use configured host/port/key and accept `--server` and `--api-key`.
Batch endpoints wait for completion; only prefix ingestion is a background job. Set
`INGESTION_DATABASE_URL` for PostgreSQL job/object persistence and startup resume. Without
it, API job/object tracking is in memory. The API uses ETag/current-pipeline tracking before
ingestion; `replace_existing` bypasses that shortcut but does not force rewriting a complete
identical passage revision. There is no transaction spanning PostgreSQL, RustFS, and Qdrant.

## Tests and troubleshooting

```bash
uv run pytest
```

Current tests simulate judge and embedding responses and verify prompt/scoring and selected
batch compatibility behavior. They do not certify live ingestion, retrieval, or model quality.

| Signal | Check |
| --- | --- |
| Collection schema error | Correct collection, dense size/cosine, learned sparse with no modifier, BM25 with IDF |
| RustFS source/credential error | Credentials, bucket/prefix, recognized extension and exactly one configured source segment |
| Embedding response error | Hybrid response including sparse vectors, dimensions, indices, finite values |
| Reranking warning | Endpoint availability and complete one-result-per-candidate response; normal RAG used RRF fallback |
| Evaluation input error | Non-empty answer/reference/question and a non-empty list of string contexts |
| Evaluation prompt error | Saved stage/raw output/retry diagnostics, judge model compliance |
| Successful command but incomplete summary | Per-record evaluation statuses; successful IDs may have been skipped |
| Repeated old answers/scores | Resume uses IDs; choose new output files for a new experiment |
