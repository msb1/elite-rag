# Benchmark workflow

Run from the repository root after ingestion. See [README](../README.md) for setup and
[operation](../docs/operation.md) for settings, resume rules, and artifact locations.
`QUESTIONS_FILE` defaults to `benchmark/questions.jsonl`.

```bash
uv run python scripts/multi_retrieval.py --input benchmark/questions.jsonl --output results/retrieval_check.jsonl
uv run python scripts/rag.py --benchmark --output results/questions_run.jsonl
uv run python scripts/eval.py --input results/questions_run.jsonl --output results/evaluation_current.jsonl
uv run python scripts/analyze.py --input results/evaluation_current.jsonl
```

Add `--limit 3` to retrieval, generation, and evaluation for a small initial run. Retrieval
output is overwritten on rerun. Generation skips saved IDs; evaluation skips successful IDs
and retries errors. Use new output filenames for a new experiment or scoring implementation.
Check evaluation statuses even if the command exits 0. Analysis output locations come from
`.env`; `--input` does not change where reports are written. The exact metric contract is
[ragas.md](../docs/ragas.md).

uv run python scripts/prepare_enterpriserag_answers.py
uv run python scripts/run_enterpriserag_eval.py --parallelism 4 --no-correction --results-file results/enterpriserag_results.json