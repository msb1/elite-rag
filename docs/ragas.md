# Implemented RAG metrics

This is the scoring contract of the current local implementation in
[`evaluation.py`](../src/elite_rag/evaluation.py), [`eval.py`](../src/elite_rag/eval.py),
[`retrieval_metrics.py`](../src/elite_rag/retrieval_metrics.py), and
[`analyze.py`](../src/elite_rag/analyze.py). The dependency is pinned to `ragas==0.4.3` in
[`pyproject.toml`](../pyproject.toml). The batch implementation ID is:

```text
ragas-0.4.3-prompts-json-parent-context-v4-structured-output
```

Elite RAG imports RAGAS prompt classes, input models, and output schemas from
`ragas.metrics.collections.*.util`. It calls their `to_string(data)` method and calculates
scores locally. It does **not** instantiate or execute the official RAGAS metric classes,
use their LLM adapter, or call `ragas.evaluate`. The formulas below describe the code as
implemented, including its empty-output and failure behavior.

## Record inputs and execution

| Elite RAG field | Meaning / use |
| --- | --- |
| `question` | Original input question (`user_input` in RAGAS terminology) |
| `answer` | Generated response (`response`) |
| `ground_truth` | Reference answer (`reference`); benchmark copies it from `gold_answer` |
| `contexts` | Ordered, deduplicated parent-document text strings used for generation |
| `expected_doc_ids`, `sources` | Used by deterministic document analysis, not by `RagEvaluator.evaluate` |
| `answer_facts` and other benchmark metadata | Preserved by benchmark/evaluation; not consumed by the five scoring methods |

`evaluate` requires `question`, `answer`, and `ground_truth` to be non-empty strings.
It does not strip these values, so whitespace-only strings pass this check. `contexts` must
be a non-empty list containing only strings; empty individual strings are allowed.
An invalid record raises `EvaluationError` with stage `input_validation` before model calls.
In particular, an answer generated with zero retrieved passages cannot pass evaluation.

Metrics run sequentially in this order: faithfulness, answer relevancy, context recall,
context precision, answer correctness. Any failure aborts the record. Earlier locally
calculated scores are not returned as a partial metrics result. Each score must be finite;
there is no clipping or validation that it lies in [0, 1].

## Judge prompts, parsing, and retries

Each RAGAS prompt includes its instruction, examples, expected output schema, and serialized
input through `to_string(data)`. `ask` adds this exact system message:

```text
Follow the user's evaluation instructions exactly. Return only one valid JSON object matching the requested response schema. Populate the schema's fields with the result; do not return the schema definition, markdown, or any extra text.
```

Requests use `EVAL_MODEL`, temperature 0, the system message, one user message, and strict
Structured Outputs (`response_format.type=json_schema`). The JSON Schema is generated from the
Pydantic output model for that specific RAGAS prompt; correction retries use the same schema.
Every object in the schema disallows extra properties. The OpenAI client uses `EVAL_BASE_URL`
(falling back to `OPENAI_LOCAL_ENDPOINT`), `EVAL_API_KEY`, and a 180-second timeout. The
configured endpoint must support OpenAI Structured Outputs; unsupported schemas fail with a
judge request error rather than silently falling back to unconstrained generation. Structured
decoding constrains shape, while Pydantic validation and the correction retry remain in place
for invalid or semantically inadequate results.

The parser scans each opening `{` in the response, uses `JSONDecoder.raw_decode` to decode
an object starting there, and returns the first candidate that validates against the RAGAS
Pydantic output model. Surrounding prose, Markdown fences, and trailing text are tolerated.
It is not a greedy brace regular expression. Schema validation alone does not establish
semantic validity of the judge's classifications or that every statement was classified.

If no candidate validates, the wrapper records a retry diagnostic and makes one additional
request with the original prompt plus an instruction to return a populated JSON instance.
The correction includes at most 1,000 characters of the first validation error; it does not
include the previous answer as an assistant message. A second invalid response raises
`EvaluationError` with both raw responses. Request failures are annotated at their prompt
stage; a request failure does not trigger this JSON correction path.

## Faithfulness

1. `StatementGeneratorPrompt` receives `{question, answer}` and returns answer statements.
2. `NLIStatementPrompt` receives `{context, statements}`, where `context` is
   `"\n".join(contexts)`. It returns statements, reasons, and verdicts.
3. For the **returned verdict list** `V`, calculate:

```text
faithfulness = sum(bool(v.verdict) for v in V) / len(V)
```

The denominator is the number of NLI outputs, not necessarily the number of originally
extracted statements. Empty generated statements return NaN, which `evaluate` rejects.
An empty NLI verdict list divides by zero and becomes a `faithfulness` stage error.
Nonzero verdicts count as true. The wrapper does not enforce binary labels beyond the
imported output schema. Reference answers and expected document IDs are not used.

## Answer relevancy

`AnswerRelevancePrompt` receives only `{response: answer}` three times, sequentially.
Each call returns `question` and `noncommittal`. Keep only results with a truthy question
string. If none remain, return 0 without embedding.

`HybridEmbedder.embed_queries` embeds `[original_question, *generated_questions]` in one
request. Only its dense vectors enter scoring. For each retained generated question:

```text
cosine(x, y) = dot(x, y) / (norm(x) * norm(y))
answer_relevancy = mean(cosine(original, generated_i))
                   * int(not all(retained_noncommittal_flags))
```

The score becomes zero only when **all retained** generated results are noncommittal.
A mixed set does not apply a fractional penalty. There is no zero-norm guard here;
a zero vector produces an error even if the noncommittal multiplier would have been zero.
Cosines are not clipped, so negative scores are possible. Embedding request failures carry
stage `answer_relevancy.embeddings`; subsequent arithmetic failures carry
`answer_relevancy`.

## Context recall

`ContextRecallPrompt` receives:

```text
question = question
context = "\n".join(contexts)
answer = ground_truth
```

It returns reference-statement classifications with reasons and integer `attributed` values.
For returned classifications `C`:

```text
context_recall = sum(c.attributed for c in C) / len(C)
```

Empty classifications return NaN and fail evaluation. Integers are summed directly, without
boolean conversion or local binary-range enforcement. The denominator is the number of
returned classifications. This measures support for statements the judge extracts from the
reference answer, not exact document retrieval, separately supplied `answer_facts`, or
support for the generated response.

## Context precision with reference

For **each entire saved context**, in its saved order, `ContextPrecisionPrompt` receives
`{question, context: that_context, answer: ground_truth}`. It returns a reason and integer
verdict `v_i`. The local code computes ranked average precision:

```text
prefix_relevant_i = sum(v_j for j in 1..i)
context_precision = sum((prefix_relevant_i / i) * v_i for i in 1..n)
                    / (sum(v_i for i in 1..n) + 1e-10)
```

Verdicts are summed and multiplied directly, without boolean conversion. With expected
binary labels, useful early contexts improve the score. This is not the useful-sentence
fraction or simply useful chunks divided by all chunks. All-zero verdicts return 0. Calling
this method directly with no contexts returns 0; the public `evaluate` path rejects that
input before calling it. For `[1, 0, 1]`, the result is `(1 + 2/3) / (2 + 1e-10)`.

## Answer correctness

1. `CorrectnessStatementPrompt` (the imported statement generator for correctness) extracts
   statements separately from `{question, answer}` and `{question, answer: ground_truth}`.
2. If both lists are non-empty, `CorrectnessClassifierPrompt` receives
   `{question, answer: answer_statements, ground_truth: reference_statements}` and returns
   lists `TP`, `FP`, and `FN`, each containing statements and reasons.
3. Let `tp`, `fp`, and `fn` be those list lengths. Calculate:

```text
precision = tp / (tp + fp) if tp + fp != 0 else (1 if fn == 0 else 0)
recall    = tp / (tp + fn) if tp + fn != 0 else (1 if fp == 0 else 0)
factuality = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
```

If either generated statement list is empty, classification is skipped and factuality is
set to **1.0**. If classification returns three empty lists, precision, recall, and
factuality are also 1.0. These are implementation edge cases, not proof of a correct answer.

`HybridEmbedder.embed_documents([answer, ground_truth])` supplies dense vectors for similarity:

```text
similarity = cosine(answer_vector, reference_vector) if both norms are nonzero else 0
answer_correctness = 0.75 * factuality + 0.25 * similarity
```

Weights and beta (F1, beta=1) are fixed in this wrapper. Similarity is not clipped or
normalized to [0, 1]. Embedding request failures use `answer_correctness.embeddings`;
arithmetic failures use `answer_correctness`.

## Embedding contract and call counts

Both embedding-backed metrics use `HybridEmbedder`, configured by `EMBEDDING_MODEL`,
`EMBEDDING_DIMENSION`, `EMBEDDING_API_KEY`, and the base URL alias
`OPENAI_REMOTE_ENDPOINT` / `EMBEDDING_BASE_URL` (the former takes precedence when both exist).
The endpoint is `<base>/embeddings`. Query and document embedding methods send the same
shape, `{model, input: [...]}`, without task prefixes or `input_type`.

Although evaluation uses only dense vectors, the client requires a valid `sparse_vector`
in every response item as well. It validates dimensions, finite numeric vector values,
unique input indices, and sparse index/value structure. It does not require nonzero norms.

For `n` contexts, non-empty extracted statements and generated questions, and successful
responses, the evaluator makes `n + 9` judge requests and two embedding requests. Prompt
validation retries can add one judge request per prompt. The extracted answer statements
for faithfulness are not reused by correctness. Evaluation never retrieves or reranks text.

## Batch persistence and errors

`run_evaluation` copies input metadata and adds the five metric fields on success, together
with `evaluation_status: "ok"` and `evaluation_implementation`. Failures record:

- `evaluation_status: "error"` and `evaluation_error`;
- `evaluation_stage` identifying validation, prompt, embedding, metric, or unknown failure;
- `evaluation_raw_responses` when attached to the failure;
- `evaluation_prompt_retries` when any initial prompt output needed correction, including
  corrections that eventually succeeded. Each item records stage, attempt=1, validation
  error, and the initial raw response.

Input and existing output are deduplicated by stringified `question_id`; the latest row wins
while first-seen order is preserved. A missing ID uses the zero-based parsed-row index.
Successful existing IDs are skipped. Errors are retried and replaced. `--limit` counts new
attempts, including failures. An existing successful row with a different or missing
implementation ID rejects the destination, even if unrelated to this input. Old error rows
alone do not trigger that rejection.

After each attempt, output is rewritten through a sibling `.tmp` file and replaced. This is
not append-only output. Rows already in the destination but absent from this input are
retained. Skipping uses ID/status, not a hash of answer content or model settings: use a new
file for a changed experiment. The runner returns exit code 0 even if records have errors.

No partial metric dictionary is returned after failure, but fields already present on an
input row are copied; judge summaries exclude error rows by status. The single-record
`/v1/evaluate` API returns only scores on success, or an HTTP error; it does not save the
batch diagnostic record.

## Deterministic document retrieval metrics

Analysis independently compares `expected_doc_ids` with parent document IDs in the saved
**final generation context**. Child hits are expanded to parent text and deduplicated by
parent document ID before generation. For older saved rows that still contain duplicate child
sources, analysis applies the same deduplication. Let `E` be the expected ID set, `P` the
ordered list of unique retrieved parent IDs, and `M = E ∩ set(P)`:

| Output | Exact definition |
| --- | --- |
| `document_recall` | `len(M) / len(E)` |
| `document_precision` | `len(M) / len(P)`, or 0 if P is empty |
| `target_doc_hit_rate` | 1 if any expected parent ID is present, else 0 |
| `all_expected_found` | 1 if every expected ID is present, else 0 |
| `reciprocal_rank` | `1 / first_relevant_document_rank`, or 0 if none |

Expected and retrieved parent ID lists remove duplicates in first-seen order. Repeated child
hits from one document do not consume additional ranks. `at_k` for k=1,3,5 applies these
formulas to the first up to k unique parent documents. A short or empty result remains in the
aggregate denominator; an empty result is a hit-rate and recall miss. `document_precision` is
the share of unique retrieved parents that are expected.

With no expected IDs, status is `no_expected_documents` and the row is excluded. With a
missing/non-list `sources`, or any source without a document ID, status is
`missing_document_ids` and the row is excluded. An empty source list is valid and scores
zero coverage for a non-empty expected set. `source_context_count` records saved source rows;
`unique_document_count` records the deduplicated parent count. The function does not verify
that the length of `sources` equals the length of `contexts`.

`summarize_document_retrieval` averages per-question scores (macro averaging), regardless of
judge status. It counts missing labels/IDs, no expected ID found, some expected IDs missing,
and expected counts exceeding retrieved parent counts. Breakdowns include `by_question_type` and
`by_source_types`; multi-source questions participate in several groups. Mean reciprocal
rank is the aggregate `reciprocal_rank`. `judge_recall_comparison` splits successful judged
rows by whether any expected ID was found and reports count, mean context recall, and count
with recall below 0.30.

These labels can be incomplete: an unlisted document can still be useful. Document presence
also does not establish passage-level fact coverage. Candidate search versus reranking loss
requires the separate retrieval test; saved answer records contain final parent contexts and
matched child-passage metadata.

## Final analysis contract

`run_analysis` uses the latest row per question ID (rows without IDs remain distinct).
`input_records` counts those rows; `records` counts rows with `evaluation_status == "ok"`.
Only successful rows contribute to the five metric summaries: count, mean, min, max, and
sample standard deviation (0 for one value, null for none). The SVG shows those means.
Document metrics include all eligible rows, including judge failures.

The hallucination export selects successful rows with `faithfulness < 0.30`; the retrieval
miss export selects successful rows with `context_recall < 0.30`. These are strict cutoffs
and heuristic review lists, not verified hallucination or retrieval diagnoses. Missing
metric fields default to 1 for these selections. Per-question document details are written
beside `ANALYSIS_OUTPUT_FILE` using `<stem>_document_retrieval.jsonl`.

Analysis does not rescore, call services, validate implementation IDs, or produce a narrative
LLM report. Do not combine historical custom scores or different implementations into a
single comparison mean. For `info_not_found` questions, the reference may be an abstention
rubric, making context recall unsuitable as a source-fact coverage measure.
