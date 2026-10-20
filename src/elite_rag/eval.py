"""Score generated RAG answers with the five custom text-judge metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from elite_rag.config import Settings
from elite_rag.evaluation import RagEvaluator


def _records(path: Path) -> list[dict[str, Any]]:
    result = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            result.append(row)
    return result


def run_evaluation(
    settings: Settings,
    *,
    input_path: str | None = None,
    output_path: str | None = None,
    limit: int | None = None,
) -> int:
    source = Path(input_path or settings.dataset_file)
    destination = Path(output_path or settings.eval_output_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = _records(source)
    completed: set[str] = set()
    if destination.exists():
        for row in _records(destination):
            if row.get("evaluation_status") == "ok" and isinstance(row.get("question_id"), str):
                completed.add(row["question_id"])
    evaluator = RagEvaluator(settings)
    count = 0
    with destination.open("a", encoding="utf-8") as handle:
        for row in rows:
            identifier = str(row.get("question_id", count))
            if identifier in completed:
                continue
            if limit is not None and count >= limit:
                break
            result = dict(row)
            try:
                result.update(evaluator.evaluate(row))
                result["evaluation_status"] = "ok"
            except Exception as exc:
                result["evaluation_status"] = "error"
                result["evaluation_error"] = str(exc)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            count += 1
            print(f"{identifier}: {result['evaluation_status']}")
    print(f"Evaluated {count} records; metrics: {destination}")
    return 0
