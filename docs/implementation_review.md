# Implementation Review Against the Design Brief

## Conformance summary

| Requested capability | Implementation | Status |
| --- | --- | --- |
| Polymorphic ingestion factory | canonical `RawDocument` plus `ParserRegistry` | Implemented |
| Natural/topological boundaries | CommonMark AST, HTML headings, messages, replies, turns, declared fields | Implemented |
| Zero token overlap | source text assigned to one child; no sliding windows | Implemented |
| 150-token children | tokenizer-aware complete vector input ceiling | Implemented |
| 2,000-token parents | tokenizer-aware parent ceiling; long documents form several parents | Implemented |
| Forced metadata injection | title, source, project, date, section precede child text | Implemented |
| Child vectors in Qdrant | named normalized EmbeddingGemma dense vectors plus miniCOIL sparse vectors | Implemented |
| Native hybrid retrieval | Qdrant dense/sparse prefetches with native RRF fusion | Implemented |
| Parent payload/docstore | parent context stored directly in each child payload | Implemented in payload mode |
| Native Qdrant filters | indexed source, project, date, document, and parent fields | Implemented |
| Parent deduplication | ordered collapse by parent ID | Implemented |
| Jina cross-encoder | asynchronous official rerank endpoint | Implemented when key is configured |
| Confidence fallback | one simplified-query search with wider limit | Implemented |
| Qwen/Llama prompt structure | escaped XML evidence and adjacent negative constraint | Implemented |
| Answer generation | `AsyncOpenAI` through FastAPI | Implemented |
| EnterpriseRAG-Bench support | all nine source families and evaluator-compatible output | Implemented |
| Remote corpus storage | RustFS/S3 uploader and remote ingestion iterator | Implemented |

## Review-driven updates

This review produced the following changes:

- Replaced line-regex Markdown heading detection with CommonMark AST tokens. This avoids
  false sections inside fenced code and supports setext headings.
- Added `LLM_TOP_P=0.1` to match the requested extraction-oriented generation profile.
- Added structured ingestion reports with seen, ingested, skipped, parent, child, batch,
  and replacement counts.
- Added FastAPI/OpenAPI operations for collection management, ingestion, retrieval, and scoring
  for mutating API workflows.
- Added `query --show-sources` for evidence IDs and scores on stderr.
- Allowed aliases to expand to multiple sources; `ticket` now retains both Jira and
  Linear candidates.
- Removed explicit `source:` and `project:` syntax from corrective embedding text while
  preserving the corresponding hard Qdrant filter.
- Added a remote FastEmbed miniCOIL sparse-vector service, collection-level IDF configuration,
  and native dense/sparse RRF fusion for primary and corrective retrieval passes.
- Added this documentation set and linked it from the project README.
- Added direct-to-RustFS archive upload and remote-object ingestion so the corpus need not
  be stored on the local computer.

## Deliberate adaptations

The brief's sample code used `https://jina.ai` as the rerank URL; the implementation uses
the API endpoint `https://api.jina.ai/v1/rerank`. It also uses Qdrant's current
`query_points` interface rather than the older `search` call.

The complete parent is not always the complete source document. Enforcing the stated
2,000-token parent ceiling requires long documents to produce multiple non-overlapping
parents. This prevents an unbounded payload and reranker input while retaining structural
context.

The code does not claim that `temperature=0.0` makes model output absolutely deterministic.
Inference backend, model version, quantization, and parallel kernels can still affect
results.

## Remaining production integrations

The code is complete at the application layer, but a live deployment still needs:

- model-license acceptance and model artifact provisioning;
- a running or managed Qdrant instance;
- a Jina credential, unless vector-only degradation is acceptable;
- a loaded OpenAI-compatible Qwen or Llama server;
- gateway authentication/authorization and tenant-derived filter enforcement;
- Qdrant backup, monitoring, and capacity planning;
- threshold calibration against representative benchmark results.
