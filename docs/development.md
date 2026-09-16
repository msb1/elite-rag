# Development and Verification

## Quality commands

```bash
uv run ruff format --check src tests
uv run ruff check src tests
uv run mypy src
uv run pytest -q
```

Tests cover declared-field loading, Markdown/HTML/email/Slack topology, fenced-code
handling, token ceilings, zero-overlap ordering, stable IDs, alias expansion, fallback
normalization, parent deduplication, corrective-pass bounds, reranker outage behavior,
prompt escaping, Qdrant payloads, and native filters.

## Adding a source type

1. Convert the API response to `RawDocument` with a stable ID and rich metadata.
2. Implement `TopologicalParser.parse` only if declared fields are not sufficient.
3. Register the parser in `default_registry`.
4. Add fixtures for normal, empty, malformed, and oversized records.
5. Assert that every source word appears in ordered child content exactly once.
6. Run a limited real-corpus ingestion before a full backfill.

Keep parsers responsible only for structure. Token sizing belongs in
`HierarchicalChunker`; embeddings belong in `EmbeddingGemma`; persistence belongs in
`QdrantVectorStore`.

## Changing chunk policy

Changing token limits or structural rules can change parent/child boundaries while IDs
remain ordinal. Rebuild a new collection for controlled comparisons. Do not mix vectors
created by different embedding models or materially different preprocessing policies in
one collection.

## Benchmark discipline

Record the git revision, environment configuration excluding secrets, embedding model,
reranker model, generation model, and Qdrant collection for every benchmark run. Tune on
a development subset and reserve held-out questions for final evaluation.

