"""Score generated answers with the five official RAGAS 0.4.3 metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from elite_rag.config import Settings
from elite_rag.evaluation import EVALUATION_IMPLEMENTATION, EvaluationError, RagEvaluator


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


def _identifier(row: dict[str, Any], index: int) -> str:
    return str(row.get("question_id", index))


def _deduplicate(rows: list[dict[str, Any]]) -> tuple[list[str], dict[str, dict[str, Any]]]:
    order: list[str] = []
    unique: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        identifier = _identifier(row, index)
        if identifier not in unique:
            order.append(identifier)
        unique[identifier] = {**row, "question_id": identifier}
    return order, unique


def _write_results(
    destination: Path, order: list[str], records: dict[str, dict[str, Any]]
) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for identifier in order:
            handle.write(json.dumps(records[identifier], ensure_ascii=False) + "\n")
        handle.flush()
    temporary.replace(destination)


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
    input_order, rows_by_id = _deduplicate(rows)
    rows = [rows_by_id[identifier] for identifier in input_order]
    output_order: list[str] = []
    output_by_id: dict[str, dict[str, Any]] = {}
    if destination.exists():
        previous_rows = _records(destination)
        for index, row in enumerate(previous_rows):
            if (
                row.get("evaluation_status") == "ok"
                and row.get("evaluation_implementation") != EVALUATION_IMPLEMENTATION
            ):
                raise ValueError(
                    f"{destination} contains a different evaluation implementation. "
                    "Use --output with a new file to keep the evaluations separate."
                )
            identifier = _identifier(row, index)
            if identifier not in output_by_id:
                output_order.append(identifier)
            output_by_id[identifier] = {**row, "question_id": identifier}
        if len(output_order) != len(previous_rows):
            _write_results(destination, output_order, output_by_id)
    completed = {
        identifier
        for identifier, row in output_by_id.items()
        if row.get("evaluation_status") == "ok"
    }
    evaluator = RagEvaluator(settings)
    count = 0
    for row in rows:
        identifier = str(row["question_id"])
        if identifier in completed:
            continue
        if limit is not None and count >= limit:
            break
        result = dict(row)
        result["evaluation_implementation"] = EVALUATION_IMPLEMENTATION
        for key in (
            "evaluation_error",
            "evaluation_stage",
            "evaluation_raw_responses",
            "evaluation_prompt_retries",
        ):
            result.pop(key, None)
        try:
            result.update(evaluator.evaluate(row))
            result["evaluation_status"] = "ok"
            completed.add(identifier)
        except EvaluationError as exc:
            result["evaluation_status"] = "error"
            result["evaluation_error"] = str(exc)
            result["evaluation_stage"] = exc.stage
            if exc.raw_responses:
                result["evaluation_raw_responses"] = exc.raw_responses
        except Exception as exc:
            result["evaluation_status"] = "error"
            result["evaluation_stage"] = "unknown"
            result["evaluation_error"] = str(exc)
        if evaluator.prompt_retries:
            result["evaluation_prompt_retries"] = evaluator.prompt_retries
        if identifier not in output_by_id:
            output_order.append(identifier)
        output_by_id[identifier] = result
        _write_results(destination, output_order, output_by_id)
        count += 1
        print(f"{identifier}: {result['evaluation_status']}")
    print(f"Evaluated {count} records; metrics: {destination}")
    return 0
