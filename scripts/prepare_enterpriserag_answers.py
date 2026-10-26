#!/usr/bin/env python3
"""Convert Elite RAG evaluation JSONL to EnterpriseRAG-Bench answer JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/evaluation.jsonl"))
    parser.add_argument("--questions", type=Path, default=Path("benchmark/questions.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/enterpriserag_answers.jsonl"))
    args = parser.parse_args()

    questions = read_jsonl(args.questions)
    expected_ids = {str(row["question_id"]) for row in questions}
    rows = read_jsonl(args.input)
    answers: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        question_id = row.get("question_id")
        answer = row.get("answer")
        sources = row.get("sources")
        if not isinstance(question_id, str) or question_id not in expected_ids:
            raise ValueError(f"{args.input}: row {index} has an unknown question_id")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"{question_id}: answer is empty or invalid")
        if not isinstance(sources, list):
            raise ValueError(f"{question_id}: sources must be a list")
        document_ids = []
        for source in sources:
            if not isinstance(source, dict) or not isinstance(source.get("document_id"), str):
                raise ValueError(f"{question_id}: a source is missing its document_id")
            document_id = source["document_id"]
            if document_id and document_id not in document_ids:
                document_ids.append(document_id)
        if question_id in answers:
            raise ValueError(f"{question_id}: duplicate answer row")
        answers[question_id] = {
            "question_id": question_id,
            "answer": answer,
            "document_ids": document_ids,
        }

    missing = expected_ids - answers.keys()
    extra = answers.keys() - expected_ids
    if missing or extra:
        raise ValueError(
            f"answer/question coverage mismatch: missing={len(missing)}, extra={len(extra)}"
        )
    order = [str(row["question_id"]) for row in questions]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for question_id in order:
            handle.write(json.dumps(answers[question_id], ensure_ascii=False) + "\n")
    print(f"Converted {len(order)} answers: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
