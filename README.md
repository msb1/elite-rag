# Elite RAG

Elite RAG indexes enterprise documents from RustFS/S3 into Qdrant, retrieves child passages
through three vector channels, reranks them with BGE, expands and deduplicates hits to parent
documents, and generates answers using an OpenAI-compatible model. It also runs the JSONL
question benchmark, evaluates saved answers, and summarizes answer quality and parent-document
retrieval success.

The active pipeline is:

```text
RustFS → parse fields/sections → child chunks → BGE-M3 dense + learned sparse / Qdrant BM25
       → Qdrant → question search + RRF → rerank child hits → expand/deduplicate parents → answer
       → saved benchmark answers → judge + embeddings → metrics → analysis + SVG
```

## Repository guide

| Path | Purpose |
| --- | --- |
| `src/elite_rag/` | Application: configuration, parsing, ingestion, storage, retrieval, generation, evaluation, analysis, API and CLI |
| `scripts/` | Batch ingestion, retrieval testing, RAG, evaluation, analysis, and HTTP clients |
| `remotes/` | Separately deployed BGE-M3 embedding and BGE reranker service, with its own dependencies |
| `benchmark/questions.jsonl` | Default question set, with expected document IDs, gold answers, and benchmark metadata |
| `benchmark/extra_questions.jsonl` | Additional questions; selected only if configured or explicitly supplied to retrieval testing |
| `tests/` | Evaluator tests using simulated model and embedding responses |
| `docs/` | [Architecture](docs/architecture.md), [concepts](docs/concepts.md), and [operation](docs/operation.md) |
| [ragas.md](ragas.md) | Exact implemented metric formulas, prompt flow, validation, retries, and document metrics |
| `results/`, `logs/` | Configurable generated artifacts and API logs |
| `configs/filter_aliases.json` | Retained configuration file; the active retriever does not load it |
| `archive/` | Historical code, configuration, and results; not the active application |

## Setup and configuration

Use Python 3.10+ and `uv`. From the repository root:

```bash
uv sync --extra dev
```

Create or edit `.env` in the repository root with service endpoints, model names, and
credentials. Application settings use explicit constructor values and `.env`; inherited
shell variables are ignored. For example, use `.env` to select a collection or output file,
not a shell environment assignment before a command. See the [configuration reference](docs/operation.md#configuration).

Required services are RustFS/S3 for source ingestion, Qdrant with server-side `Qdrant/bm25`
inference, the BGE model service, and an answer model. Evaluation needs the judge and BGE
embedding services but does not query RustFS, Qdrant, or the answer model. PostgreSQL is
optional for persistent API ingestion jobs.

The BGE service has a separate Python 3.11–3.13 environment and defaults to a CUDA reranker.
On its configured host, from a checkout of this repository:

```bash
uv run --project remotes python remotes/remote-models.py
```

It reads `.env` relative to the working directory and provides `/v1/embeddings` and
`/v1/rerank`. Configure matching client endpoints locally. The scripts below invoke the
application directly and do not require the Elite RAG HTTP server.

## 1. Ingest documents

The default bucket/prefix is `s3://enterprise-rag/docs/`. `MULTI_SOURCE_TYPES` selects source
folders: by default Confluence, Fireflies, GitHub, Gmail, Google Drive, HubSpot, Jira, Linear,
and Slack. Objects must use configured source path segments and have `.json`, `.txt`, or
`.md` extensions.

```bash
uv run python scripts/ingest_multi.py --limit 10
uv run python scripts/ingest_multi.py
# Or ingest one configured source:
uv run python scripts/ingest_multi.py --source fireflies
```

The script creates or validates the collection and saves `MULTI_INGESTION_REPORT`. It embeds
section-aware passages without an LLM, publishes complete verified revisions, and skips a
revision whose full ready passage count already matches. Rerunning rechecks objects from the
beginning. Individual object failures are reported while later objects continue; inspect the
report and exit status. See [ingestion operation](docs/operation.md#ingest-documents).

## 2. Test retrieval

```bash
uv run python scripts/multi_retrieval.py --input benchmark/questions.jsonl --limit 3 --output results/retrieval_check.jsonl
uv run python scripts/multi_retrieval.py --input benchmark/questions.jsonl
```

This searches the original question across dense, learned sparse, and BM25 channels, fuses
results with Qdrant RRF, and reranks all fused candidates. It saves unique-parent-document
rankings before and after reranking, expected-document ranks, parent-document hit@5 fields, and
a `.manifest.json` beside the output. It does not generate answers or call the judge. Each
invocation overwrites its output.

`RETRIEVAL_NUMBER` limits each channel and the fused child-passage candidate set (default 50).
The reranker scores those child candidates. The RAG path maps reranked hits to parent document
IDs, deduplicates parents in first-hit order, and applies `RERANK_TOP_N` to the unique parents
(default 5). It sends each selected parent's full text to the answer model. Duplicate child
hits from one parent do not consume another parent slot. The separate
retrieval test keeps all reranked child hits and reports ranks after deduplicating by parent
document ID.

## 3. Run RAG on questions.jsonl

Ask one question:

```bash
uv run python scripts/rag.py --question "What does the indexed material say about ...?"
```

Run a small batch, then continue the full question set into the same output:

```bash
uv run python scripts/rag.py --benchmark --limit 3 --output results/questions_run.jsonl
uv run python scripts/rag.py --benchmark --output results/questions_run.jsonl
```

`QUESTIONS_FILE` selects the input and defaults to `benchmark/questions.jsonl`. The output
contains benchmark metadata plus `answer`, ordered `contexts`, corresponding `sources`,
`warnings`, and `ground_truth` copied from `gold_answer`. Already saved question IDs are
skipped; `--limit` counts new rows. Use a new output filename when comparing a changed model,
collection, or retrieval policy. `PROGRESS_FILE` is a progress marker; output question IDs
control resume. Existing answer files were generated with child-only contexts; regenerate to
a new output file to evaluate the parent-context behavior described above.

## 4. Evaluate saved answers

```bash
uv run python scripts/eval.py --input results/questions_run.jsonl --output results/evaluation_current.jsonl --limit 3
uv run python scripts/eval.py --input results/questions_run.jsonl --output results/evaluation_current.jsonl
```

The evaluator reuses RAGAS 0.4.3 prompt classes and Pydantic output schemas, sends ordinary
chat completions to the configured judge, and calculates five scores locally: faithfulness,
answer relevancy, context recall, context precision with reference, and answer correctness.
Relevancy and correctness also use the configured BGE dense embeddings. This is a local
wrapper, not execution of the official metric classes. [ragas.md](ragas.md) specifies its
exact formulas and edge cases.

The current implementation ID is `ragas-0.4.3-prompts-json-parent-context-v3`. Successful rows are
skipped; failed rows are retried and replaced. A successful row from another implementation
causes rejection of that output file: choose a new filename. Malformed judge output gets one
correction retry per prompt. Inspect `evaluation_status` and diagnostic fields: the batch
command can exit successfully even when individual evaluations fail.

## 5. Produce the final summary analysis

```bash
uv run python scripts/analyze.py --input results/evaluation_current.jsonl
```

Analysis uses saved data only. It writes the configured summary, SVG dashboard, failure
exports, and per-question document retrieval details. With code defaults these are:

| Artifact | Contents |
| --- | --- |
| `results/analysis.json` | Metric count/mean/min/max/sample standard deviation and deterministic document coverage summaries |
| `results/rag_metrics_dashboard.svg` | Mean of each of the five judge/embedding metrics |
| `results/hallucinations.jsonl` | Successful rows with faithfulness strictly below 0.30 |
| `results/retrieval_misses.jsonl` | Successful rows with judge context recall strictly below 0.30 |
| `results/analysis_document_retrieval.jsonl` | Expected, retrieved, matched, and missing document IDs per question, including judge failures |

Read `input_records` versus `records` first to see evaluation coverage. Then inspect all five
metric means and failure exports. Use `document_retrieval.at_k.5.target_doc_hit_rate` (shown
as `target_doc_hit_rate@5` in the console) as the success rate for finding at least one
expected parent document in the first five parent ranks. Its cutoffs at 1/3/5 count unique
parent-document ranks, not child passages. Document recall measures the fraction of all
expected parent IDs found; document precision is the share of returned parent IDs that are
expected. Source groups can overlap. The filename `retrieval_misses.jsonl` denotes low judge
context recall; it does not prove that an expected document was absent.

RAGAS context recall measures reference-statement support in the parent text supplied to the
answer model. Document recall measures expected parent-ID coverage in the final saved contexts.
Neither guarantees that every required
answer fact was retrieved. Compare the separate retrieval test's candidate rankings to
investigate where documents were lost. For `info_not_found` questions, a gold abstention
rubric is not a set of source facts; interpret judge recall separately. Analysis does not
call an LLM to write a narrative conclusion or reject files containing mixed implementations.

## EnterpriseRAG-Bench scoring

`results/evaluation.jsonl` contains all 500 generated answers, including rows whose separate
RAGAS evaluation failed. Convert those answers and their final-context document IDs, then run
EnterpriseRAG-Bench's `metrics_based_eval.py` using `EVAL_MODEL` and the endpoint in `.env`:

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/onyx-dot-app/EnterpriseRAG-Bench.git /private/tmp/EnterpriseRAG-Bench
git -C /private/tmp/EnterpriseRAG-Bench sparse-checkout set src answer_evaluation
uv run python scripts/prepare_enterpriserag_answers.py
uv run python scripts/run_enterpriserag_eval.py --limit 1 --parallelism 1
uv run python scripts/run_enterpriserag_eval.py --resume --parallelism 4
```

The runner adapts the benchmark's OpenAI Responses API transport to the local server's
OpenAI-compatible chat-completions endpoint; the benchmark scoring functions and prompts
remain unchanged. By default it passes `--no-correction` and skips citation stripping, then
writes `results/enterpriserag_results.json`. No benchmark corpus download is needed for this
mode. Document recall is computed against the original expected IDs. Since the evaluator has
no candidate-document judgments in this mode, every retrieved document outside the expected
set counts as an invalid extra; treat that metric as an upper-bound penalty until running
with the full document corpus and correction flow. Do not call this leaderboard-verified.

The benchmark score judges answer correctness holistically, checks each `answer_fact` for
completeness, and reports document recall and invalid extra documents. These scores differ
from Elite RAG's RAGAS metrics. For leaderboard submission, follow the benchmark's current
verification and reproducibility requirements.

## API and local tests

```bash
uv run elite-rag serve
uv run pytest
```

The API defaults to `http://127.0.0.1:8080`; schemas are at `/docs`. `API_KEY`, when non-empty,
protects `/v1/` routes with a bearer token. Routes cover collection initialization, ingestion,
retrieval, RAG, single-record evaluation, benchmark, batch evaluation, and analysis. API
startup requires Qdrant. See [operation](docs/operation.md#http-api) for routes and examples.

CLI equivalents for the quality workflow are `elite-rag benchmark`, `elite-rag eval`, and
`elite-rag analyze`; `elite-rag rag "question"` answers one question. Without explicit paths,
these commands use `.env` settings (`DATASET_FILE`, `OUTPUT_FILE`, etc.), which may differ
from the code defaults documented here.
