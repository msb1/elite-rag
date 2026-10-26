# Concepts

This page explains the active pipeline. See [architecture](architecture.md) for module and
service boundaries, [operation](operation.md) for commands, and [the metric contract](../ragas.md)
for exact scoring behavior.

## Documents, sections, and passages

A source object becomes a `SourceDocument` with a document ID, source, title, content fields,
and metadata. Parsing preserves field boundaries and recognizes Markdown/HTML structure,
headings, lists, tables, code, and common ticket/PR labels. Structured lists and dictionaries
are rendered into section text. A `Section` contains its field name, heading path, and text.

`HierarchicalPassageChunker` splits within sections. Its body maximum is
`MULTI_CHUNK_MAX_TOKENS` (default 300). Effective overlap is
`min(CHUNK_OVERLAP_TOKENS, max(0, body_max - 64))`, and its body packing target is
`max(64, body_max - overlap)` (default 260). Overlap copies trailing complete atoms that fit;
it is not a guaranteed number of repeated tokens. Long paragraphs, sentences, and code are
split further when necessary.

The active hierarchical splitter uses those body budgets rather than the generic
`CHUNK_TARGET_TOKENS` / `CHUNK_MAX_TOKENS` packing policy. The full enriched embedding text
must fit `BGE_EMBEDDING_MAX_LENGTH` (default 1024); an oversized passage raises an error.
Some generic chunk settings remain in the pipeline signature even though they do not set
the active body budget.

Each child passage stores two text forms. `text` is the parsed body used for BM25 and reranking.
`embedding_text` adds source/entity context, title, and section path for BGE dense and learned
sparse embeddings. The reranker separately constructs title + section + body. Each child also
stores its parent document ID and available full-document/section text in its payload. The
vector search indexes child passages; after reranking, the RAG path resolves those hits to
parent document IDs, deduplicates them, and expands to parent text for generation. There is
no separately searched parent-vector collection.

## Revision and readiness

A revision fingerprints title, fields, metadata, and chunking policy. Passage IDs are
stable for a document/revision/ordinal. New points have `ingestion_ready=false`; the service
verifies the expected revision count, marks it ready, verifies the ready count, then deletes
other revisions of that document. Search filters for ready points.

This protocol makes incomplete new revisions invisible and allows deterministic retries.
It is not a transaction covering all stores: there can be a brief interval between publishing
a new revision and deleting the old ready revision. A complete ready revision is skipped on
rerun. An empty parsed document removes existing points for that document ID. API object
tracking can additionally skip objects by ETag; the batch script relies on passage revisions.

## Candidates, RRF, reranking, and final context

Each question uses three Qdrant search channels:

| Channel | Representation |
| --- | --- |
| Dense | BGE-M3, 1024 dimensions, cosine distance |
| Learned sparse | BGE-M3 lexical token weights, no Qdrant modifier |
| BM25 sparse | `Qdrant/bm25` inference from passage body and raw question, with IDF |

The original question is embedded once for the first two channels. Each channel prefetches
up to `RETRIEVAL_NUMBER` hits. Qdrant RRF combines their ranks and returns at most that many
fused passage candidates. RRF scores and reranker scores are different quantities.

The BGE reranker receives every fused child-passage candidate and must return each candidate
exactly once in ranked order. `RagService.retrieve` maps the ranked child hits to parent
documents by parent document ID, deduplicates parents in first-hit order, then applies
`RERANK_TOP_N` to unique parents. It sends each selected parent's full stored text to generation.
Therefore the LLM receives up to `RERANK_TOP_N` parent documents; duplicate child hits from
one parent do not consume another parent slot. If full parent text is unavailable, selected
section text is used with a warning. On reranker failure, the same expansion runs on the RRF
child-hit order.
`/v1/retrieve` exposes those deduplicated parent contexts rather than the child candidate pool.

Retrieval success is judged on parent document IDs: a question is a target-document hit when
at least one expected parent ID is present. The separate retrieval test reranks the full fused
child candidate list, then deduplicates by parent document ID for its document ranks. The
`MULTI_DOCUMENT_SCORE` setting is validated but does not control ranking.

## Grounded answers and abstention

The answer model sees the original question and selected parent-document texts with source labels.
It is instructed to use explicit context facts, omit source labels from its prose, and say
exactly `I cannot find the answer in the provided documents.` if evidence is insufficient.
That instruction is a prompt, not a factual correctness guarantee. When no passage is found,
the service returns that sentence directly without calling the answer model.

The response includes ordered `contexts`, aligned `sources` metadata, and warnings. Benchmark
generation preserves those fields, copies the gold answer into `ground_truth`, and appends
one record per new question ID. Existing output IDs determine resume.

## Two kinds of evaluation

The five judge/embedding metrics use RAGAS 0.4.3 prompts and schemas through a local wrapper:

| Score | Question it addresses |
| --- | --- |
| Faithfulness | What fraction of returned answer-statement verdicts is supported by context? |
| Answer relevancy | How similar are questions inferred from the answer to the original question, with an all-noncommittal penalty? |
| Context recall | What fraction of the reference answer's classified statements is supported by combined context? |
| Context precision | Are contexts judged useful for the reference ranked early, using average precision? |
| Answer correctness | How well do answer/reference statements match (75% factual F1) and their embeddings agree (25% cosine)? |

The wrapper executes prompts and formulas itself; it does not run the official metric
classes. It validates record inputs and structured judge outputs, retries malformed output
once per prompt, and fails the record on a metric error. Exact formulas and edge cases,
including empty statements and zero vectors, are in [ragas.md](../ragas.md).

Deterministic document metrics compare benchmark `expected_doc_ids` with parent document IDs
in the final saved sources. They report expected-document recall and precision, target-document
hit rate (at least one expected parent found), complete coverage, and reciprocal rank. Cutoffs
at 1, 3, and 5 count unique parent document ranks; repeated child hits do not consume slots.
They run even when judge evaluation failed, provided expected labels and parent IDs are valid.

Document presence does not prove the selected passage contains the answer. An unlisted
document may also provide useful evidence. Judge context recall does not check expected IDs,
and a low judge score is not proof of a document retrieval miss. Questions with no expected
IDs are excluded from document metrics. For `info_not_found`, assess abstention separately
because the reference may describe expected behavior rather than source facts.

## Analysis and comparison

Analysis summarizes saved scores, exports low-faithfulness/low-context-recall rows, and
computes document coverage without calling models. Judge means use only successful records;
document means use all eligible records. It keeps the latest row per question ID. Source
group summaries overlap for multi-source questions.

Keep output files separate for changes to models, retrieval settings, collections, or
scoring implementations. Generation/evaluation resume by question ID, not by an experiment
hash. Analysis does not enforce implementation consistency. Compare evaluation coverage and
question mix before interpreting changes in means.
