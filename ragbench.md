# Comparing Elite RAG with EnterpriseRAG-Bench

This guide explains what the EnterpriseRAG-Bench evaluator produces and how to run it yourself against the answers already saved by Elite RAG. The evaluator makes LLM judge calls. Running the commands below is an explicit action for you to take in your terminal; this guide does not run them.

## Inputs and preparation

Elite RAG saves generated answers in `results/evaluation.jsonl`. The preparation script matches each row's `question_id` against `benchmark/questions.jsonl`, and converts the answer and retrieved source document IDs into the input format expected by EnterpriseRAG-Bench:

```text
results/enterpriserag_answers.jsonl
```

Each prepared row contains `question_id`, `answer`, and `document_ids` from the saved parent-document sources. Preparation checks that every benchmark question has exactly one answer, that answers are non-empty, and that source document IDs are present. It does not call an LLM. Answer files generated before parent-context expansion contain child contexts; regenerate answers to a new output file before evaluating the updated retrieval behavior.

From the repository root, prepare the benchmark input with:

```bash
uv run python scripts/prepare_enterpriserag_answers.py
```

The default paths can be changed with `--input`, `--questions`, and `--output`.

## Run the evaluator

The runner expects an EnterpriseRAG-Bench checkout at `/private/tmp/EnterpriseRAG-Bench` by default, or a different checkout supplied with `--bench-repo`. It reads the evaluation model and endpoint from Elite RAG's `.env` settings. The adapter sends judge requests using OpenAI-compatible chat completions and leaves the benchmark's scoring code and prompts in place.

Run the full set from the Elite RAG repository root:

```bash
uv run python scripts/run_enterpriserag_eval.py --parallelism 4
```

This command invokes the LLM judge. By default, the runner uses `results/enterpriserag_answers.jsonl` as input and writes `results/enterpriserag_results.json`. It passes `--no-correction` and `--skip-citation-stripping` to the benchmark evaluator. `--parallelism` controls how many questions are evaluated concurrently; it defaults to 4. It is a throughput setting, not a scoring setting.

Other options include:

```text
--bench-repo PATH       EnterpriseRAG-Bench checkout
--answers-file PATH     Prepared answers JSONL
--questions-file PATH   Benchmark questions JSONL
--results-file PATH     Results JSON output
--limit N               Evaluate only the first N questions
--parallelism N         Number of concurrent question evaluations
--resume                Reuse already-scored questions in the results file
```

Use `--resume` to continue an interrupted run from the same results file. The evaluator updates that file as questions finish, so it can contain a partial snapshot while a run is in progress. Without `--resume`, do not expect an existing results file to be continued; use a new output path if you want to preserve a prior run.

## What appears after evaluation

The evaluator prints progress as individual questions finish. At the end, it prints a summary in the terminal with the number of scored questions, skipped rows, corrected questions, average correctness, average completeness, combined correctness-completeness score, average document recall, and average invalid extra documents.

The same evaluation is saved as structured JSON in `results/enterpriserag_results.json`. Its main sections are:

- `aggregate_stats`: overall counts and metric averages for the questions scored so far.
- `question_type_stats`: the same aggregate metrics grouped by benchmark question type.
- `questions`: one result per scored question, sorted by question ID. Rows include `question_id`, `question_type`, `answer_correct`, `correctness_reasoning`, `completeness_pct`, `document_recall_pct`, `invalid_extra_docs`, and `corrected`.

The headline `combined_correctness_completeness_score` is the mean completeness percentage, with completeness set to zero for any answer judged incorrect. `average_correctness_pct` is the percentage judged correct. `average_completeness_pct` is the mean fact completeness across answers. `average_recall_pct` is document ID recall against the benchmark's expected document IDs, averaged over questions that have expected documents. `average_invalid_extra_docs` is the mean count of retrieved documents judged invalid, also averaged over applicable questions.

The JSON is a set of per-question results plus aggregates; it is not just one scalar score. For a complete run, confirm `aggregate_stats.completed_questions` matches `aggregate_stats.total_questions`. If not, treat the file as a partial run and resume or investigate skipped/error rows before comparing it.

## Interpretation and comparison limits

This runner forces `--no-correction`. It does not load the benchmark's full document corpus or run the three-judge document correction flow. As a result, the evaluator cannot identify retrieved documents that are valid but absent from the original expected document IDs; such extra IDs are counted as invalid. Interpret the invalid-extra-documents metric as a conservative penalty in this mode.

These are EnterpriseRAG-Bench evaluator metrics produced using the configured judge model. They are not automatically an official or leaderboard-verified score. Compare runs only when the question set, answer coverage, evaluator options, and judge setup are recorded and comparable. For an official leaderboard submission, follow the benchmark maintainers' current verification and reproducibility process.

## Files

- Source answers: `results/evaluation.jsonl` (input; preparation does not overwrite it)
- Prepared benchmark answers: `results/enterpriserag_answers.jsonl`
- Benchmark evaluation output: `results/enterpriserag_results.json`
- Input conversion: `scripts/prepare_enterpriserag_answers.py`
- Evaluator adapter: `scripts/run_enterpriserag_eval.py`

The prepared answers and evaluator output are generated files. Keep or remove them according to whether you need to inspect or resume a run; do not remove the source `results/evaluation.jsonl` if you still need those saved answers.
