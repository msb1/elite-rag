# Configuration Reference

Settings are loaded from process environment variables, then the local `.env`. Unknown
variables are ignored. Secrets belong only in `.env` or a secret manager; `.env` is
gitignored. The environment variable names match the working sibling `rag-bench`
project.

## Qdrant

| Variable | Default | Meaning |
| --- | --- | --- |
| `QDRANT_URL` | `http://192.168.1.50:6333` | REST endpoint; matches the sibling `rag-bench` project |
| `QDRANT_API_KEY` | empty | optional managed-cluster credential |
| `QDRANT_COLLECTION` | `elite_rag` | collection containing child points |
| `QDRANT_DENSE_VECTOR_NAME` | `dense` | named EmbeddingGemma vector field |
| `QDRANT_SPARSE_VECTOR_NAME` | `sparse` | named miniCOIL sparse vector field |
| `QDRANT_UPSERT_MAX_ATTEMPTS` | `5` | initial upsert plus bounded transient-failure retries |
| `QDRANT_UPSERT_INITIAL_BACKOFF_SECONDS` | `0.5` | first retry delay; later delays double |
| `QDRANT_UPSERT_MAX_BACKOFF_SECONDS` | `4.0` | cap for retry delay |
| `INGESTION_DATABASE_URL` | `postgresql://user:password@192.168.1.50:5432/elite_rag` | durable ingestion-run and object-work-item ledger |
| `INGESTION_LEASE_SECONDS` | `900` | object lease duration before a stopped worker's work can be reclaimed |
| `INGESTION_PIPELINE_VERSION` | `v1` | checkpoint compatibility version; change after a material pipeline change |
| `SPARSE_EMBEDDING_MODEL` | `Qdrant/minicoil-v1` | miniCOIL model identity, deployed by the remote sparse service |
| `SPARSE_EMBEDDING_URL` | `http://192.168.1.50:8000/v1/embeddings/sparse` | remote miniCOIL sparse-vector endpoint |
| `SPARSE_EMBEDDING_BATCH_SIZE` | `8` | maximum texts sent to the remote sparse service in one request |
| `SPARSE_EMBEDDING_TIMEOUT_SECONDS` | `30` | remote sparse-service request timeout |
| `HYBRID_CANDIDATE_LIMIT` | `100` | dense and sparse prefetch window before native RRF fusion |

The collection stores a named dense cosine vector and a named miniCOIL sparse vector with
Qdrant's `IDF` modifier. Changing vector names, the embedding dimension, or the collection
schema requires a compatible collection. Never recreate a production collection without a
backup and a replayable source corpus.

Elite RAG uses the standard `qdrant-client`; it does not install or execute FastEmbed. Deploy the
[combined remote models service](../services/remote-models.md) on the remote model computer. It
returns numeric sparse vectors that Elite RAG sends to Qdrant and preserves the Jina reranker at
the same host and port. Dense and sparse vectors are fused inside Qdrant with reciprocal-rank
fusion (RRF), so the application does not perform a second search or merge rankings in Python.

Each Qdrant upsert is retried only for transport errors and Qdrant HTTP 5xx responses. The
default five attempts use 0.5, 1, 2, and 4 second delays after failed attempts. Every retry
reuses the same deterministic point IDs, so it is safe if Qdrant applied a write before the
connection failed. After the final attempt, the ingestion job fails as an infrastructure failure;
the document is not recorded in `MISSING_DOCS_FILE`.

Prefix ingestion uses PostgreSQL as its source of operational truth. It records one durable run
and one work item per RustFS object. Completed objects are skipped when a failed prefix request is
submitted again with the same source, collection, and pipeline version. An interrupted object is
reclaimed when its lease expires; deterministic Qdrant IDs make reprocessing it safe.

## Embedding and chunking

| Variable | Default | Meaning |
| --- | --- | --- |
| `OPENAI_LOCAL_ENDPOINT` | `http://192.168.1.50:1234/v1` | LM Studio-compatible API endpoint |
| `EMBEDDING_MODEL` | `text-embedding-embeddinggemma-300m` | EmbeddingGemma server model ID |
| `EMBEDDING_DIMENSION` | `768` | Qdrant vector dimension |
| `CHILD_MAX_TOKENS` | `150` | complete metadata-enriched child input ceiling |
| `PARENT_MAX_TOKENS` | `2000` | complete parent context ceiling |
| `INGEST_BATCH_SIZE` | `64` | child vectors embedded and upserted per batch |
| `LOG_FILE` | `logs/elite-rag.log` | append-only application and exception log |
| `LOG_LEVEL` | `INFO` | application log verbosity |
| `MISSING_DOCS_FILE` | `logs/missing_docs.json` | JSON list of documents skipped for document-specific failures |

Embedding and generation both use the local OpenAI-compatible endpoint and the exact
server-visible model identifiers from `rag-bench`.

## RustFS object storage

| Variable | Default | Meaning |
| --- | --- | --- |
| `S3_ENDPOINT_URL` | `http://192.168.1.50:9000` | RustFS S3-compatible endpoint |
| `S3_ACCESS_KEY` | empty | RustFS access key |
| `S3_SECRET_KEY` | empty | RustFS secret key |
| `S3_BUCKET` | `enterprise-rag` | destination bucket |
| `S3_PATH_MARKDOWN_PREFIX` | `documents` | object prefix preserving source folders |

The RustFS uploader uses path-style S3 addressing and multipart uploads. Archive mode
streams ZIP members over HTTPS directly into RustFS. `ingest-rustfs` streams each remote
object into Elite RAG; no corpus copy is created under `data/`.

Token ceilings use the loaded embedding model's tokenizer. Lower child sizes improve
semantic precision but increase point and payload counts. Larger parents improve distant
context but increase reranker input, generation context, and repeated Qdrant payload.

## Retrieval and reranking

| Variable | Default | Meaning |
| --- | --- | --- |
| `JINA_RERANK_URL` | `http://192.168.1.50:8000/v1/rerank` | local rerank API endpoint |
| `JINA_RERANK_MODEL` | `jina-reranker-v3.5` | local cross-encoder model name |
| `RERANK_TOP_N` | `5` | parent contexts returned by reranking |
| `RERANK_CONFIDENCE_THRESHOLD` | `0.35` | best-score trigger for corrective pass |
| `PRIMARY_VECTOR_LIMIT` | `25` | first-pass child candidates |
| `FALLBACK_VECTOR_LIMIT` | `50` | corrective-pass child candidates |

The confidence threshold is empirical, not universal. Tune it on representative held-out
questions and pin the reranker model while comparing runs.

Aliases live in `configs/filter_aliases.json`. Each alias can map to one canonical string
or a list. For example, `ticket` maps to both Jira and Linear so the hard filter does not
discard one ticket system.

## Generation

| Variable | Default | Meaning |
| --- | --- | --- |
| `RAG_MODEL` | `qwen2.5-7b-instruct-mlx` | Qwen server-visible model identifier |
| `EVAL_MODEL` | `meta-llama-3.1-8b-instruct` | Llama 3.1 evaluator model identifier |
| `LLM_API_KEY` | `lm-studio` | local API placeholder |
| `LLM_MAX_TOKENS` | `1024` | answer token ceiling |
| `LLM_TEMPERATURE` | `0.0` | low-variance sampling temperature |
| `LLM_TOP_P` | `0.1` | nucleus sampling bound |

Model identifiers must match the model names exposed by the local LM Studio-compatible
server.
