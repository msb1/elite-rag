# Theory of Operation

## Operating principles

Elite RAG follows six invariants:

1. Structure precedes token count. Natural document boundaries are discovered before any
   size enforcement.
2. Source text does not overlap between child chunks. Recall comes from parent expansion
   and deeper fallback search, not duplicate windows.
3. Retrieval metadata participates twice: compact metadata is embedded into child text,
   while rich metadata remains independently filterable in Qdrant.
4. Dense and sparse child vectors jointly select evidence; parent text is what the reranker and
   generator read.
5. Retrieval is bounded and inspectable. One primary pass and no more than one corrective
   pass are allowed.
6. Missing evidence produces an explicit abstention instruction rather than an invitation
   to complete from model memory.

## Ingestion lifecycle

### 1. Normalize

A loader converts a file or API record into `RawDocument`. Stable upstream IDs should be
used whenever possible. The source name is normalized to lowercase. Rich scalar, list,
and nested metadata is preserved in the eventual payload.

### 2. Parse topology

The registry emits ordered structural blocks. Parsers remove structural headers from the
body and place them in a hierarchy path. This prevents header text from being treated as
unrelated content while still making the path available to embeddings and parent context.

### 3. Build hierarchy

The chunker computes budgets with the embedding model's tokenizer. Parent grouping and
child splitting are separate operations:

- parents preserve enough neighboring evidence for distant facts and reranking;
- children remain narrow enough for high-resolution dense and sparse matching;
- a child points to exactly one parent;
- a parent belongs to exactly one source document.

When a single sentence exceeds a budget, whitespace units are used first. If a URL, hash,
identifier, or code-like unit has no whitespace and exceeds the budget, the chunker uses a
tokenizer-aware character split as the final lossless fallback. Formatting whitespace may be
normalized, but source characters are neither copied to adjacent chunks nor silently discarded.

### 4. Embed and persist

The ingestion engine batches child inputs, embeds them, and upserts Qdrant points. Normal
reruns overwrite deterministic IDs. The API's `replace_existing` option first deletes all points for a
document, which also removes stale tail chunks after a document becomes shorter. Because
that mode is delete-then-upsert, operators should use it only when source data can be
replayed after interruption.

## Query lifecycle

```text
query
  -> parse source/project intent
  -> dense EmbeddingGemma + sparse miniCOIL query representations
  -> filtered Qdrant dense/sparse prefetches
  -> native Qdrant RRF fusion
  -> hybrid candidate window, then outer retrieval limit
  -> collapse unique parents
  -> rerank parent text
       | score >= threshold -> return
       | score < threshold  -> simplify query
       |                      -> wider filtered hybrid child search
       |                      -> collapse + rerank -> return
       | Jina unavailable    -> preserve vector order -> return
  -> compile evidence prompt
  -> stream generation tokens
```

Both prefetches use the same active metadata filters and Qdrant fuses their rankings with RRF.
The fallback keeps active metadata filters. An explicit user constraint is therefore
never silently relaxed. If an organization uses implicit aliases that are too broad or
too narrow, adjust `configs/filter_aliases.json` rather than allowing semantic search to
override authorization-like boundaries.

## Confidence semantics

`RERANK_CONFIDENCE_THRESHOLD` is a routing threshold, not a probability of correctness.
Its correct value depends on corpus language, document length, reranker version, and
question distribution. Calibrate it against held-out benchmark questions. A score below
the threshold triggers more recall; it does not by itself prove that the corpus lacks an
answer.

If Jina is disabled or raises a transport/response error, the system returns the primary
vector order and does not spend latency on the corrective pass. If Jina responds normally
with low scores, the corrective pass remains enabled.

## Generation and abstention

Parents are escaped before insertion into XML. Each context includes source, modification
date, and document ID attributes. The model must cite document indexes. When no retrieved
context supports an answer, the prompt directs the model to emit the fixed abstention
sentence.

Low sampling parameters make extraction more stable, but grounding still depends on
retrieval quality and model instruction following. Applications should retain retrieved
document IDs alongside answers for audit and evaluation.

## Failure behavior

| Failure | Behavior |
| --- | --- |
| Malformed benchmark field declarations | affected document is recorded and skipped when the loader can identify the object |
| Oversized whitespace-free token | token-aware character splitting preserves the document and chunk budget |
| Empty parsed document | skipped and counted in the ingestion report |
| Embedding or Qdrant failure | current operation fails; deterministic IDs permit replay |
| No primary Qdrant matches | corrective hybrid pass runs with the same hard filters |
| Jina transport or response failure | vector-ranked parents are returned |
| Local LLM failure before streaming | caller receives the OpenAI client error |
| LLM failure after streaming begins | stream terminates; clients must treat it as partial |

## Non-goals

- No multi-query generation or additional rank-merging stage.
- No token-window overlap.
- No hidden LLM-based intent classification.
- No claims that low temperature guarantees factuality.
- No authorization policy engine; upstream identity and access controls must determine
  which metadata filters a user is allowed to apply.
