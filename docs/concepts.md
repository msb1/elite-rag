# Concepts

This page explains the data and retrieval terms used by the implementation.

## Documents, sections, passages

A source record is normalized to a `SourceDocument`: a stable document ID, source name,
title, structured content fields, and metadata. Parsing preserves field boundaries and
recognizes Markdown and HTML headings, tables, lists, code blocks, and common ticket or pull
request labels. The resulting `Section` carries the original field name, a heading path, and
its text.

The hierarchical chunker divides section text into passages sized for the embedding model.
It aims for `CHUNK_TARGET_TOKENS`, respects `CHUNK_MAX_TOKENS`, and carries up to
`CHUNK_OVERLAP_TOKENS` from the preceding passage where possible. It keeps paragraph and code
block boundaries in mind, splitting long text further when needed. Each passage stores its
body and section-aware embedding text. Embedding text includes source and document context
such as title, project, repository, issue, or space metadata when available.

Passage payloads keep the original body for generation, plus the document ID, source, title,
field name, section path, offsets, revision, token count, and pipeline signature. This makes
retrieval results traceable to their source material.

## Revisions and ready passages

A revision is a content fingerprint scoped to the ingestion pipeline signature. The
signature changes when relevant settings such as embedding model, tokenizer, chunk limits,
or pipeline version change. Stable passage IDs make retrying an interrupted write idempotent.

`ingestion_ready` is a publication marker in Qdrant payload. New or rewritten passages remain
false until the full revision has been written and verified. Search only sees ready passages.
After the new revision is ready, older revisions for the same document are deleted. This
avoids mixing passages from different versions in normal retrieval.

## Three retrieval channels

Elite RAG searches three representations of each passage:

- **Dense:** semantic similarity from the 1024-dimensional BGE-M3 vector.
- **Learned sparse:** BGE-M3 lexical weights, useful for meaningful terms and phrases.
- **BM25:** lexical scoring computed by Qdrant from the original passage body, using the
  `Qdrant/bm25` model and IDF modifier.

For a question, the embedding client returns dense and learned sparse query vectors. Qdrant
also receives the raw question for BM25 inference. Each channel produces candidates and Qdrant
combines their rankings with reciprocal rank fusion (RRF). RRF combines ranks rather than
requiring scores from different channels to share a scale. The configured retrieval count
sets the candidate limit for each channel and the fused result.

## Reranking and grounded generation

The BGE reranker scores each candidate against the original question using title, section
path, and passage body. The best `RERANK_TOP_N` passages become the answer context. If
reranking fails, the service records a warning and retains the fused RRF order.

The answer model is instructed to use only retrieved context and to return a fixed
no-answer sentence when the context is insufficient. The response contains the answer,
passage contexts, source identifiers and metadata, and any retrieval warnings. It does not
ask the model to emit source labels in its prose.

## Evaluation terms

The benchmark records questions, expected document IDs, gold answers, and answer facts.
Generated answers and retrieved contexts are scored by an LLM text judge with five custom
metrics:

- **Faithfulness:** share of answer claims supported by context.
- **Answer relevancy:** semantic alignment between the input question and a question inferred
  from the answer.
- **Context recall:** share of gold facts found or deducible from context.
- **Context precision:** share of context sentences judged useful for the question.
- **Answer correctness:** fact-level precision and recall against the gold answer, combined
  as an F1-style score.

These are model-judged estimates used to compare runs; they are not deterministic proofs of
correctness.
