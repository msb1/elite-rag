# Elite RAG

Standalone Enterprise RAG application for documents stored in the `enterprise-rag` RustFS
bucket. It uses hierarchical chunking, tri-vector Qdrant retrieval, BGE reranking, grounded
answer generation, and the project's JSONL question benchmark.

## Configuration and services

`.env` is the application's configuration source. The default collection is `elite_rag_v2`,
the RustFS bucket is `enterprise-rag`, and the document prefix is `docs`. Configure the
Qdrant, RustFS, embedding, reranking, answer-model, and judge-model endpoints there. The
implementation, retrieval concepts, and operating procedures are documented in
[`docs/architecture.md`](docs/architecture.md), [`docs/concepts.md`](docs/concepts.md), and
[`docs/operation.md`](docs/operation.md).

The BGE model service in `remotes/remote-models.py` exposes `/v1/embeddings` for dense and
learned sparse BGE-M3 vectors and `/v1/rerank` for the BGE reranker. The local ingestion and
retrieval clients call those endpoints. The service is source agnostic: it embeds chunks from
Confluence, Jira, GitHub, Fireflies, Gmail, Google Drive, HubSpot, Linear, and Slack.

Start the model service on its host:

```bash
uv run --project remotes python remotes/remote-models.py
```

Start the RAG API:

```bash
uv run elite-rag serve
```

The default API address is `http://127.0.0.1:8080`. Set `SERVER_HOST`, `SERVER_PORT`, and
optionally `API_KEY` in `.env`.

## Enterprise document ingestion

`scripts/ingest_multi.py` reads Markdown and structured source records from
`s3://enterprise-rag/docs/<source>/` for every source in `MULTI_SOURCE_TYPES`. The default
configuration includes all nine source folders:

`confluence`, `jira`, `github`, `fireflies`, `gmail`, `google_drive`, `hubspot`, `linear`, and
`slack`.

Each chunk retains document ID, title, section path and name, source metadata, and full document
text. Chunks are section aware with a nominal 300-token body limit. Ingestion does not call an
LLM. It requests dense and learned sparse BGE-M3 vectors from the model server and submits the
BM25 text to Qdrant for server-side inference with the IDF modifier.

Ingest all configured source folders:

```bash
uv run python scripts/ingest_multi.py
```

Ingest one source or limit a run:

```bash
uv run python scripts/ingest_multi.py --source fireflies
uv run python scripts/ingest_multi.py --limit 10
```

Unchanged documents are skipped using their content revision, so restarting an interrupted
ingestion safely rechecks from the beginning and continues past already indexed documents.

## Retrieval and answer generation

The collection has three named vectors: 1024-dimensional BGE-M3 dense, BGE-M3 learned sparse,
and Qdrant server-side BM25 sparse with IDF. Each query is embedded by the BGE server; Qdrant
searches all three streams and fuses them with RRF. `RETRIEVAL_NUMBER` controls the fused
candidate count, and all candidates are sent to the BGE reranker.

Run retrieval evaluation over `benchmark/questions.jsonl`:

```bash
uv run python scripts/multi_retrieval.py
```

Ask a single question with the full RAG pipeline:

```bash
uv run python scripts/rag.py --question "What does the indexed material say about ...?"
```

The full pipeline searches the user's question once across the three vector channels, fuses
those channel results with Qdrant RRF, reranks the candidates with BGE, and generates a grounded
answer from the top `RERANK_TOP_N` passages.

## Benchmark and evaluation

The enterprise benchmark input is `benchmark/questions.jsonl`. It includes each question,
source type, expected document IDs, gold answer, and answer facts. Run a small batch or the full
set:

```bash
uv run python scripts/rag.py --benchmark --limit 3
uv run python scripts/rag.py --benchmark
```

Generation resumes by skipping question IDs already present in `DATASET_FILE`. Evaluate the
answers with five custom text-judge metrics:

```bash
uv run python scripts/eval.py --limit 3
uv run python scripts/analyze.py
```

Metrics are faithfulness, answer relevancy, context recall, context precision, and answer
correctness. Results and analysis artifacts are written under `results/`.

The HTTP API exposes `/v1/retrieve`, `/v1/rag`, `/v1/evaluate`, `/v1/benchmark`, `/v1/eval`,
`/v1/analyze`, and ingestion and collection endpoints. Request and response schemas are
available at `/docs`.

## Archive

The previous implementation and its retrieval outputs are preserved in
`archive/pre_full_reimplementation_2026-10-16/`.
