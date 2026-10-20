uv run python scripts/rag.py --benchmark --output results/questions_run.jsonl
uv run python scripts/eval.py --input results/questions_run.jsonl --output results/evaluation_run.jsonl
uv run python scripts/analyze.py --input results/evaluation_run.jsonl