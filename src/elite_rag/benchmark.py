"""Run the Enterprise RAG question set through the local RAG pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from elite_rag.config import Settings
from elite_rag.rag import RagService


def _load_questions(path: Path) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict) or not isinstance(row.get("question"), str):
                raise ValueError(f"{path}:{line_number}: expected a question object")
            questions.append(row)
    return questions


def run_benchmark(
    settings: Settings,
    *,
    limit: int | None = None,
    question_id: str | None = None,
    output_path: str | None = None,
) -> int:
    questions = _load_questions(Path(settings.questions_file))
    output = Path(output_path or settings.dataset_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    progress_path = Path(settings.progress_file)
    processed: set[str] = set()
    if output.exists():
        with output.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                identifier = record.get("question_id")
                if isinstance(identifier, str):
                    processed.add(identifier)

    service = RagService(settings)
    completed = 0
    try:
        with output.open("a", encoding="utf-8") as handle:
            for index, source_row in enumerate(questions):
                identifier = str(source_row.get("question_id", index))
                if question_id is not None and identifier != question_id:
                    continue
                if identifier in processed:
                    continue
                if limit is not None and completed >= limit:
                    break
                question = str(source_row["question"]).strip()
                if not question:
                    continue
                result = service.answer(question)
                row = {
                    **source_row,
                    "question_id": identifier,
                    "answer": result["answer"],
                    "contexts": result["contexts"],
                    "sources": result["sources"],
                    "warnings": result["warnings"],
                    "ground_truth": str(source_row.get("gold_answer", "")),
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                temporary = progress_path.with_suffix(progress_path.suffix + ".tmp")
                temporary.parent.mkdir(parents=True, exist_ok=True)
                temporary.write_text(
                    json.dumps({"last": index, "question_id": identifier}) + "\n",
                    encoding="utf-8",
                )
                temporary.replace(progress_path)
                completed += 1
                print(f"{identifier} ({completed}): {question}")
                print(result["answer"])
                if result["warnings"]:
                    print("Warnings: " + "; ".join(result["warnings"]))
    finally:
        service.close()
    print(f"Completed {completed} questions; results: {output}")
    return 0
