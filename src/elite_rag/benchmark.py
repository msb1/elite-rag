from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from elite_rag.runtime import Runtime


async def run_benchmark(
    runtime: Runtime,
    questions_path: Path,
    output_path: Path,
    *,
    limit: int | None = None,
    resume: bool = False,
) -> int:
    completed = _completed_ids(output_path) if resume else set()
    processed = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume else "w"
    with (
        questions_path.open(encoding="utf-8") as source,
        output_path.open(mode, encoding="utf-8") as output,
    ):
        for line in source:
            if limit is not None and processed >= limit:
                break
            question: dict[str, Any] = json.loads(line)
            question_id = str(question["question_id"])
            if question_id in completed:
                continue
            query = str(question["question"])
            contexts = await runtime.retrieval.retrieve(query)
            answer = await runtime.generation.generate(query, contexts)
            document_ids = list(dict.fromkeys(context.document_id for context in contexts))
            output.write(
                json.dumps(
                    {
                        "question_id": question_id,
                        "question": query,
                        "answer": answer,
                        "ground_truth": str(question.get("gold_answer", "")),
                        "contexts": [context.text for context in contexts],
                        "document_ids": document_ids,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            output.flush()
            processed += 1
            print(f"[{processed}] {question_id}: {len(document_ids)} documents", flush=True)
            await asyncio.sleep(0)
    return processed


def _completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as handle:
        return {str(json.loads(line)["question_id"]) for line in handle if line.strip()}
