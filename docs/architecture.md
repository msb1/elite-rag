# Elite RAG Architecture

## Purpose

Elite RAG is a retrieval-augmented generation service for heterogeneous corporate
knowledge. Its design target is the Onyx EnterpriseRAG-Bench corpus, while its parser
registry and canonical document model allow API records and standalone files to use the
same pipeline.

Retrieval has one primary dense pass and, when the cross-encoder score is below a
configured threshold, one corrective dense pass using a deterministic query reduction.

## System context

```text
Corporate APIs and exported files
          |
          v
  Canonical RawDocument
          |
          v
  Polymorphic parser registry
          |
          v
 Ordered TopologicalBlocks
          |
          v
 Hierarchical zero-overlap chunker
       /             \
      v               v
  child text      parent context
      |               |
 metadata injection   |
      |               |
 EmbeddingGemma       |
      \               /
       Qdrant point payload
              |
     deterministic filter
              |
       dense child search
              |
       unique parent collapse
              |
          Jina rerank
              |
       grounded XML prompt
              |
  Qwen/Llama-compatible stream
```

## Components

### Canonical documents

`RawDocument` separates identity, source, title, content fields, and metadata. The
EnterpriseRAG-Bench loader obtains title and content fields from each JSON record's
`title_field_name` and `content_field_names`. This is important because source schemas
are intentionally different and some benchmark documents are deliberately misfiled.

### Parser registry

The registry selects a parser from the canonical source name:

| Sources | Parser boundary |
| --- | --- |
| Markdown, blog, Google Drive | CommonMark AST headings; fenced code is not mistaken for a heading |
| Confluence | HTML heading nodes and text descendants |
| Gmail/email | current message and quoted-reply markers |
| Slack | blank-line message boundaries |
| Fireflies/transcript | timestamped speaker turns |
| Jira, Linear, GitHub, HubSpot, unknown | declared content fields |

Every parser emits an ordered list of `TopologicalBlock(path, text)`. A source-specific
parser falls back to declared fields when the expected native field is absent.

### Parent-child chunking

The chunker first reserves token space for document metadata and section labels. It then
splits oversized blocks in this order: paragraph, sentence/line, and finally words for an
otherwise indivisible oversized atom. Adjacent units are packed up to configured limits.

- Child vector input defaults to at most 150 model tokens.
- Parent context defaults to at most 2,000 model tokens, including its header.
- Source text is assigned to one child only. There is no sibling text overlap.
- Structural labels and metadata intentionally repeat; “zero overlap” refers to source
  content, not retrieval metadata.
- Documents longer than 2,000 tokens produce multiple parents. Small documents normally
  produce one parent containing the complete document.

UUIDv5 identifiers derived from document ID, hierarchy level, and ordinal make repeated
ingestion idempotent as long as document ordering and content boundaries are stable.

### Embeddings and Qdrant

EmbeddingGemma receives asymmetric inputs:

```text
title: <title> | text: source: <source> | project: <project> |
updated: <YYYY-MM> | section: <path> | content: <child text>
```

Queries receive:

```text
task: search result | query: <user query>
```

Vectors are normalized before cosine search. Each Qdrant point contains one child vector
and this payload:

```json
{
  "document_id": "dsid_...",
  "parent_id": "uuid-v5",
  "child_content": "SECTION: ...",
  "vectorized_content": "title: ... | text: ...",
  "parent_context": "DOCUMENT ID: ...",
  "metadata_filters": {"source": "jira", "project_id": "..."}
}
```

The full parent is repeated on its child points. This gives a self-contained Qdrant
deployment and constant-time parent reconstruction at the cost of payload storage. A
future external docstore can replace this duplication without changing retrieval APIs.

### Retrieval and reranking

The deterministic intent router recognizes explicit `source:` and `project:` filters and
organization-defined aliases. Qdrant applies source and project constraints before dense
search. Multiple values within one field use OR; source and project fields use AND.

Primary retrieval requests 25 child points by default. Matches collapse by `parent_id`,
preserving the best vector rank for each parent. Jina reranks distinct parent strings and
returns the configured top N. If the best reranker score is below the confidence
threshold, filler words and explicit filter clauses are removed and a single 50-point
corrective search runs. No additional query fan-out or rank-merging stage occurs.

### Grounded generation

The prompt factory serializes evidence as escaped `<Document>` elements with stable `D1`,
`D2`, and subsequent labels. The negative no-evidence instruction appears immediately
before the user query. Document content is explicitly treated as evidence rather than
instructions to reduce prompt-injection risk.

The generation client uses an OpenAI-compatible streaming API with `temperature=0.0` and
`top_p=0.1` defaults. These settings reduce sampling variance but cannot guarantee
bit-for-bit determinism across different inference engines, hardware, or model versions.

## Deployment boundaries

The FastAPI service owns orchestration only. Qdrant, Jina, Hugging Face model artifacts,
and the local generation server remain external dependencies. The HTTP service
construct the same runtime graph, avoiding divergent retrieval behavior.
