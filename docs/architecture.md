# Architecture

Elite RAG is a Python service and command-line application for indexing enterprise content,
retrieving relevant passages, and generating answers grounded in those passages. Qdrant is the
retrieval and passage store. RustFS is the object source for batch ingestion. Model inference
runs through separately configured embedding, reranking, answer, and evaluation endpoints.

This document describes the implementation in `src/elite_rag` and the batch utilities in
`scripts`. The commands are documented in [operation.md](operation.md), the terms in
[concepts.md](concepts.md), local metric behavior in [ragas.md](ragas.md), and the
EnterpriseRAG-Bench workflow in [ragbench.md](ragbench.md).
The high-level flow is:

```mermaid
flowchart LR
    subgraph Inputs[Content inputs]
        RF[RustFS objects<br/>JSON, Markdown, text]
        DD[Direct document API]
    end
    subgraph Ingestion[Ingestion path]
        RD[Object reader]
        N[Source normalization]
        SE[Section extraction]
        HC[Hierarchical passage chunker]
        EM[Embedding client]
        IC[Ingestion coordinator]
        BI[Batch ingestion script]
    end
    subgraph Models[Model endpoints]
        EMB[BGE-M3 embedding endpoint]
        RER[BGE reranking endpoint]
        LLM[Answer model endpoint]
        JUDGE[Evaluation judge endpoint]
    end
    Q[(Qdrant<br/>tri-vector collection)]
    API[FastAPI application]
    CLI[CLI and scripts]
    RAG[RAG service]
    EVAL[Benchmark, evaluation<br/>and analysis]

    RF --> RD --> N --> SE --> HC
    DD --> API --> N
    HC --> IC
    IC --> EM
    EM <--> EMB
    IC --> Q
    BI --> RD
    CLI --> BI
    API --> IC
    API --> RAG
    CLI --> RAG
    RAG --> EM
    RAG --> Q
    RAG --> RER
    RAG --> LLM
    API --> EVAL
    CLI --> EVAL
    EVAL --> RAG
    EVAL <--> JUDGE
    EVAL --> EMB
    EVAL --> Files[JSONL and analysis artifacts]
```

The diagram groups modules by responsibility. The API and CLI are entry points; they use the
same ingestion and RAG services where applicable. The API prefix-ingestion job and batch ingestion script list RustFS objects. Direct API
ingestion accepts one document payload, while `scripts/ingest_multi.py` performs an
independent sequential batch over configured source folders.

## Runtime components

| Component | Implementation | Responsibility |
| --- | --- | --- |
| HTTP application | `api.py` | FastAPI lifecycle, authentication middleware, endpoint validation, error mapping, and thread-offloaded work |
| Settings | `config.py` | `.env` loading, defaults, configuration validation, and ingestion pipeline signature |
| Source reader | `sources.py` | S3-compatible RustFS object listing, metadata lookup, byte reads, and source resolution from object key |
| Normalization and parsing | `parsing.py`, `documents.py` | Convert raw records to `SourceDocument`; emit heading-aware `Section` values |
| Chunking and tokens | `chunking.py`, `tokenization.py` | Preserve section context, split to token budgets, generate passage metadata and stable IDs |
| Embedding | `embedding.py` | Call the BGE-M3 embeddings API and validate response vectors |
| Collection and writes | `multi.py`, `store.py` | Enforce the tri-vector collection schema, index payload fields, write vectors and perform revision operations |
| Ingestion orchestration | `ingestion.py` | Batch embedding, passage upserts, verification, readiness commit, and replacement cleanup |
| Prefix job control | `jobs.py` | Queue, process, persist, resume, and report API RustFS prefix jobs |
| Retrieval and answer | `rag.py` | Three-channel search, RRF, BGE reranking, context assembly, and grounded answer generation |
| Quality workflow | `benchmark.py`, `evaluation.py`, `eval.py`, `retrieval_metrics.py`, `analyze.py` | Run benchmark questions, ask judge model for scores, summarize results, and export failure rows |
| EnterpriseRAG-Bench adapter | `scripts/prepare_enterpriserag_answers.py`, `scripts/run_enterpriserag_eval.py` | Convert saved answers and source IDs, then invoke an external benchmark checkout using the configured judge |
| Command line | `cli.py`, `scripts/` | Start the API and run RAG, benchmark, evaluation, analysis, ingestion, and retrieval utilities |

The package's `PassageStore` contains reusable Qdrant collection and revision operations.
The active application uses `MultiPassageStore`, which extends those operations with the
learned sparse and server-side BM25 vectors. The older two-vector schema helper remains in
`store.py`, but active tri-vector ingestion and retrieval validate and use `multi.py`.

## Ingestion architecture

### Input adapters and normalized document shape

`RustFSReader` reads `.json`, `.txt`, and `.md` objects. JSON records use declared
`title_field_name` and `content_field_names`; those fields become the title and content, and
remaining values become metadata. Text and Markdown use the first non-empty line as the title
and the remaining text as content. The object key and resolved source are retained. Source
names are inferred from configured path segments; an object key must contain exactly one
configured source segment, and an explicitly requested source must agree with it.

The API's direct-document endpoint creates the same `SourceDocument` form from the submitted
ID, content, source, title, and metadata. This keeps downstream parsing and indexing common
across source adapters.

### Section extraction and passage formation

The parser preserves structured field names. Within a field it recognizes Markdown heading
hierarchies, HTML headings and block elements, HTML and Markdown tables, lists, and fenced
code. It can split common source labels such as Jira's “Steps to reproduce” or GitHub's
“Testing” into additional section levels. Each section is represented by a field name, path,
and text.

The hierarchical chunker operates within sections to avoid crossing heading boundaries.
`MULTI_CHUNK_MAX_TOKENS` caps bodies (default 300); the body target is the maximum minus
effective overlap (default 260), with a minimum of 64. `BGE_EMBEDDING_MAX_LENGTH` caps
enriched embedding text (default 1024). Generic `CHUNK_TARGET_TOKENS` and `CHUNK_MAX_TOKENS`
are not this splitter's active body budgets. It uses the configured tokenizer; long
paragraphs or sentences are split further, while code fences are kept as blocks until they must be divided.
Passages may include a tail overlap from the previous passage in the same section. The
embedding text adds source and available entity context (for example project, repository,
issue, or space), title, and section path before the passage body. The original body remains
available separately for search payload and answer context.

Each child passage contains a deterministic point ID, parent document ID, ordinal, field and
section identity, parsed-section offsets where recoverable, token count, and metadata. Metadata
includes a content revision, pipeline signature, source and normalized entity attributes,
source metadata, and section evidence. The child is one Qdrant point; parent documents are
not separately vector-indexed. Payloads retain parent text assembled from non-empty string
fields, plus the full parsed section text for fallback context expansion. Structured list and
dictionary fields are parsed into passages but are omitted from `document_full_text`; if any
string-field parent text is available, generation uses that text rather than adding retrieved
structured sections to it.

### Vector generation and collection schema

The active path requires BGE-M3 and a 1024-dimension dense vector. Each chunk is sent to the
embedding endpoint in configured batches. The endpoint response supplies a dense vector and
BGE learned sparse indices and values. For the third vector, the application sends the
original passage body as a Qdrant `Document` input using `Qdrant/bm25`; Qdrant performs the
BM25 inference and applies IDF.

| Vector name (configurable) | Type and behavior | Input |
| --- | --- | --- |
| `dense` | 1024-dimensional dense vector, cosine distance | Context-enriched passage embedding text |
| `sparse` | BGE-M3 learned sparse vector, no modifier | Context-enriched passage embedding text |
| `sparse_bm25` | Qdrant BM25 sparse vector, IDF modifier | Original passage body, inferred by Qdrant |

The names above are defaults. `multi_settings()` requires all three names to be non-empty and
distinct, requires the BGE-M3 configuration, and requires the BM25 model to be
`Qdrant/bm25`. On collection creation and reuse, the store ensures payload indexes for document ID,
source, project ID, revision, field name, section ID, and readiness. On reuse, it validates
the dense size/distance, sparse vector presence, and modifiers; incompatible schemas fail with a
collection schema error.

### Revision commit protocol

Revision identity combines a hash of document title, fields, metadata, and pipeline policy.
Point IDs are stable for the document, revision, and passage ordinal. Upserts mark points as
`ingestion_ready=false`, so partial work cannot enter normal retrieval. The coordinator then
checks the revision point count, marks every point in the revision ready, checks the ready
count, and removes other revisions for that document.

```mermaid
sequenceDiagram
    participant Caller as API job or batch script
    participant Ingest as IngestionService
    participant Embed as BGE-M3 endpoint
    participant Store as MultiPassageStore
    participant Q as Qdrant

    Caller->>Ingest: ingest(SourceDocument)
    Ingest->>Store: ensure / validate collection
    Ingest->>Ingest: parse, chunk, compute revision
    Ingest->>Store: count ready points for revision
    alt full revision already ready
        Store-->>Ingest: expected count matches
        Ingest->>Store: delete other revisions
        Ingest-->>Caller: unchanged report
    else revision needs writing
        loop embedding batches
            Ingest->>Embed: embed passage texts
            Embed-->>Ingest: dense + learned sparse vectors
            Ingest->>Store: upsert points with ready=false
            Store->>Q: dense + learned sparse + BM25 Document points
        end
        Ingest->>Store: verify expected point count
        Store->>Q: mark revision ready=true
        Ingest->>Store: verify ready point count
        Ingest->>Store: delete older revisions
        Ingest-->>Caller: ingested report
    end
```

Until readiness is published, new points are excluded from retrieval. If the ready count
does not match after publication, `commit_revision` attempts to reset the revision to not
ready and raises an error. Publication is not an atomic revision swap: new and old ready revisions may
coexist briefly before cleanup. A subsequent ingestion attempt can upsert the same
deterministic IDs and retry.
An unchanged revision is skipped only when the expected count of ready points is present. An
empty parsed document deletes all points with that document ID. `replace_existing` is passed
through as report and job behavior; actual passage replacement is governed by revision
comparison and the Qdrant write protocol.

### Batch and asynchronous ingestion

`scripts/ingest_multi.py` lists configured source folders under the selected RustFS prefix,
then processes objects sequentially. It writes a JSON report after each object, records
failures, and continues to later objects when an individual read or ingestion fails. Qdrant
revision checks allow safe reruns even if the script stops partway through.

The API prefix endpoint creates an ingestion job and returns its ID. `IngestionJobs` runs its
work in a background thread, reads RustFS objects, and uses object ETags for a fast unchanged
check before parsing. Tracking is scoped by bucket, object key, collection, and pipeline
signature. With `INGESTION_DATABASE_URL`, job state and object tracking are stored
in PostgreSQL; startup resumes jobs recorded as queued or running, and advisory locks prevent
multiple workers from processing the same resumed job concurrently. Without a database URL,
job and object state lives in process memory and does not survive restart. The API also
supports one RustFS object and direct document ingestion synchronously through the shared
ingestion service.

Prefix jobs skip invalid document content and count it as failed. Other read or ingestion
errors fail the job; startup resumes only jobs still recorded as queued or running. A failed
job requires a new submission, which can reuse object tracking and ready revisions.

## Retrieval and answer architecture

### Retrieval, reranking, and parent context assembly

`TriVectorRetriever` validates the configured collection at construction, embeds a question
once for dense and learned sparse query vectors, then sends three Qdrant prefetch queries:
dense, learned sparse, and BM25 with the original question text. All three are filtered to
`ingestion_ready=true`. Each prefetch and the fused result use `RETRIEVAL_NUMBER` as their
limit. Qdrant combines the ranked candidates with reciprocal rank fusion (RRF). The client
discards hits with missing or blank passage text.

`RagService.retrieve` passes the original question and each child candidate's title, section
path, and text to the BGE reranking endpoint. A valid response must rank every candidate
exactly once. If reranking fails, the service retains the Qdrant fused order and adds a warning.
No query expansion or query rewriting is performed.

`expand_to_parent_documents` then groups child hits by parent document ID in first-hit order
and selects up to `RERANK_TOP_N` unique parents. Hits without a parent ID are skipped with a
warning. Expansion uses text already present in the retrieved payloads; it makes no additional
Qdrant or RustFS read. Stored `document_full_text` takes precedence. When it is unavailable,
the service joins retrieved sections in section order, falling back to child text when section
text is absent, and adds a warning. Parents with no usable text are skipped. The resulting
context includes title, source, matched sections, and parent text.

The returned provenance includes matched passage IDs and sections collected during this scan.
The scan stops at the first new parent beyond the cutoff, so it may not collect later child
matches for already selected parents. `RERANK_TOP_N=5` limits generation to at most five unique
parent contexts. `MULTI_DOCUMENT_SCORE` is validated but does not control this ordering.

```mermaid
sequenceDiagram
    participant Caller as API, CLI, or benchmark
    participant RAG as RagService.retrieve
    participant Embed as BGE-M3 embedding endpoint
    participant Q as Qdrant
    participant Rank as BGE reranker

    Caller->>RAG: original question
    RAG->>Embed: embed query once
    Embed-->>RAG: dense + learned sparse query vectors
    RAG->>Q: dense + learned sparse + BM25 prefetch, then RRF
    Note over RAG,Q: Each prefetch filters ingestion_ready=true
    Q-->>RAG: fused child passages and payloads
    alt candidates remain after text validation
        RAG->>Rank: question + title/section/body for every candidate
        alt complete ranking returned
            Rank-->>RAG: ordered indexes + relevance scores
        else reranking fails
            Rank-->>RAG: error or invalid response
            Note over RAG: Keep RRF order and add warning
        end
        RAG->>RAG: select unique parents up to RERANK_TOP_N
        RAG->>RAG: expand payload text and collect source metadata
    end
    RAG-->>Caller: parent contexts + warnings
```

`POST /v1/retrieve` returns these parent contexts under the JSON key `passages` and stops
here. `RagService.answer`, used by `POST /v1/rag`, the RAG CLI, and benchmark generation,
continues through the following flow.

### Answer generation after retrieval

`RagService.answer` receives the parent contexts and warnings from `retrieve`. With no
contexts it immediately returns `I cannot find the answer in the provided documents.` with
empty `contexts` and `sources` lists. Otherwise it wraps each context in a numbered
`Source ID: [Doc N]` block and joins the blocks into the user message under
`[Retrieved Context]`, followed by the original question under `[User Query]`.

```mermaid
flowchart TD
    Retrieved["Parent contexts, source metadata, and warnings from retrieve"] --> HasContext{"Any parent context?"}
    HasContext -->|No| Abstain["Fixed no-answer text; empty contexts and sources"]
    HasContext -->|Yes| Blocks["Wrap ordered contexts in Source ID: Doc N blocks"]
    Blocks --> UserMessage["User message: Retrieved Context + User Query"]
    Question["Original question"] --> UserMessage
    UserMessage --> Chat["Answer model: chat.completions.create"]
    System["System instruction: use context facts; abstain if insufficient; omit source labels"] --> Chat
    Config["RAG_MODEL, RAG_TEMPERATURE, RAG_MAX_TOKENS"] --> Chat
    Chat --> Content["Read first choice content and strip whitespace"]
    Chat -->|Request failure| Error["Raise error; RAG API maps it to HTTP 502"]
    Content --> Nonempty{"Non-empty answer?"}
    Nonempty -->|No| Error
    Nonempty -->|Yes| Result["Return question, answer, contexts, sources, warnings"]
    Abstain --> Result
    Retrieved -.->|Aligned context text and source metadata| Result
```

The request uses `RAG_SYSTEM_PROMPT` as the system message and the assembled context/question
as a single user message. It calls the OpenAI-compatible endpoint configured by
`OPENAI_LOCAL_ENDPOINT`, with `LLM_API_KEY`, `RAG_MODEL`, `RAG_TEMPERATURE`, and
`RAG_MAX_TOKENS`. The application sends the full assembled contexts without a total input-token
budget or truncation step; `RAG_MAX_TOKENS` limits generated output. Requests carry no prior
conversation history.

The system prompt asks the model to use only explicit context facts, omit source labels,
and return the fixed no-answer sentence when evidence is insufficient. This behavior is
prompt-driven: the application checks for an empty answer but performs no factual verification
or citation validation before returning it. Evaluation is a separate workflow.

For a generated answer, `contexts` contains the same ordered context strings used in the
prompt. The aligned `sources` list includes parent document IDs, the representative child
passage ID, collected matched child IDs/sections, title, source, and optional rerank score.
Reranking and expansion warnings are returned alongside the result. Model request failures
and empty answers propagate as errors rather than being converted to abstentions.

## API and process boundaries

The FastAPI lifespan configures logging, initializes the ingestion job manager and Qdrant
collection, clears object tracking if the collection had to be newly created, and resumes
unfinished persisted jobs. Expensive synchronous retrieval, generation, ingestion, benchmark,
and evaluation work is dispatched with `asyncio.to_thread` by the endpoints.

When `API_KEY` is non-empty, middleware protects paths beginning with `/v1/` by comparing the
Authorization header with the configured key after removing an optional `Bearer ` prefix.
`/health` and OpenAPI docs are outside that middleware check. External
service failures are generally logged and translated to HTTP errors; collection schema
mismatch is returned as a conflict during collection initialization.

The main endpoint groups are:

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Process health response |
| `POST /v1/retrieve` | Return ranked parent contexts under `passages`, plus warnings, without answer generation |
| `POST /v1/rag` | Retrieve, rerank, and generate a grounded answer |
| `POST /v1/collections/initialize` | Create or validate collection; optional recreation |
| `POST /v1/ingest/document` | Ingest one supplied document |
| `POST /v1/ingest/rustfs/object` | Ingest one RustFS object |
| `POST /v1/ingest/rustfs/prefix` | Submit a background RustFS prefix job |
| `GET /v1/ingest/jobs/{job_id}` | Read job status and counters |
| `POST /v1/benchmark`, `/v1/eval`, `/v1/analyze` | Run batch quality workflows |
| `POST /v1/evaluate` | Score one supplied question, answer, context set, and gold answer |

The API uses cached service instances within a process for ingestion, jobs, and RAG. Qdrant
is accessed independently by ingestion and retrieval clients. The application does not
provide a shared transaction spanning PostgreSQL, Qdrant, and RustFS; object tracking is
recorded after Qdrant ingestion succeeds, and revision readiness is managed inside Qdrant.

## Benchmark and evaluation architecture

### Answer generation and local evaluation

`benchmark.py` reads JSONL questions and sends each unprocessed question through `RagService`.
It appends results with answers, contexts, source records, warnings, and the gold answer to
the configured dataset file. It skips question IDs already present in the output and writes
a progress marker after each completed question.

`evaluation.py` uses imported RAGAS 0.4.3 prompt classes and Pydantic output schemas,
but executes ordinary chat completions and calculates the five metrics locally. It does not
execute the official metric classes. Record validation requires non-empty question, answer,
and reference strings and a non-empty string context list. Prompts execute sequentially;
malformed output gets one correction retry. A failure aborts the record without returning
partial scores. Relevancy and correctness additionally request dense vectors through the
hybrid embedding client. Exact formulas, prompt fields, and failure cases are specified in
[ragas.md](ragas.md).

`eval.py` tags rows with `ragas-0.4.3-prompts-json-parent-context-v3`. It deduplicates input/output by
question ID, skips successful IDs, replaces errors on retry, and rewrites output through a
temporary file after each attempt. A successful preexisting row from a different implementation
rejects the destination. The batch return code is 0 even if individual rows failed; consumers
must inspect `evaluation_status`. ID-based resume does not detect changed answers or models.

`analyze.py` retains the latest row per question ID and summarizes successful judge scores,
writes a mean-score dashboard SVG, and exports successful records below the strict 0.30
faithfulness/context-recall thresholds. These exports are review lists, not verified failures.
`retrieval_metrics.py` independently computes expected parent-document ID coverage over final
saved sources, including judge-error rows with valid labels and IDs. It reports recall,
precision, target-document hit rate (any expected parent found), complete coverage, reciprocal
rank, cutoffs at 1/3/5 unique parent documents, and breakdowns by question/source type. Cutoff
aggregates score the first up to k parents; short or empty result lists remain in the
denominator as misses. Analysis writes numeric summaries and per-question details without calling models or checking implementation
consistency. These deterministic document metrics are computed during analysis; `eval.py`
adds the five judge/embedding scores and their status to saved answer records.

### Retrieval diagnostics

`scripts/multi_retrieval.py` is a separate diagnostic workflow: it reranks all fused child
candidates, then saves deduplicated parent-document rankings before/after reranking,
expected-document ranks, target-document hit@5 flags, and a manifest. It neither generates answers nor applies
`RERANK_TOP_N`. Its output is overwritten per invocation; reranking failures produce error
rows instead of the normal RAG fallback. Its ranks use unique parent document IDs, matching
the final-context analysis.

### EnterpriseRAG-Bench evaluation

This separate workflow uses saved answers and final source document IDs rather than invoking
retrieval or generation again. `scripts/prepare_enterpriserag_answers.py` validates answer
coverage against the question set, rejects duplicate question IDs, and writes
`question_id`, `answer`, and deduplicated `document_ids` to the prepared JSONL. It does not
depend on successful local RAGAS evaluation or call a model.

`scripts/run_enterpriserag_eval.py` imports an external EnterpriseRAG-Bench checkout and
adapts its LLM client to streaming OpenAI-compatible chat completions using Elite RAG's
evaluation endpoint, key, and model settings. The benchmark checkout owns the scoring logic,
prompts, result aggregation, and resume behavior. The runner forwards the question/answer
paths, result path, concurrency, optional limit, and resume flag. It always enables
`--no-correction` and `--skip-citation-stripping`; the document correction flow is therefore
disabled. The benchmark writes its own JSON results, separate from the local evaluation
JSONL and analysis outputs. See [ragbench.md](ragbench.md) for the workflow and interpretation.

## Configuration and service dependencies

Settings are loaded from explicit settings values and `.env`; the settings class does not
read inherited shell variables or ambient secret files. The principal dependency boundaries
are:

| Service | Used for | Client configuration |
| --- | --- | --- |
| Qdrant | Collection, passage payloads, vector search, RRF, BM25 inference | `QDRANT_URL`, optional API key, collection and vector names |
| RustFS / S3 | Source object listing and reads | Endpoint, access key, secret, region, bucket, and prefix |
| BGE-M3 endpoint | Dense and learned sparse retrieval embeddings; dense evaluation embeddings for relevancy and correctness | Embedding base URL, model, dimension, and API key |
| BGE reranker | Candidate ordering against the question | `BGE_RERANK_URL`, timeout; model and max length are configured on the model service |
| Answer model | Final grounded answer | OpenAI-compatible base URL, model, temperature, token limit, API key |
| Evaluation judge | Benchmark metric judgments | Judge base URL, model, and API key |
| PostgreSQL (optional) | Durable API job state and object ETag tracking | `INGESTION_DATABASE_URL` |

BGE inference is implemented in the separately deployed `remotes/remote-models.py`; the
answer and judge model endpoints are external services. The defaults and operating procedures are maintained in [operation.md](operation.md).
