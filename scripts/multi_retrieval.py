#!/usr/bin/env python3
"""Evaluate direct tri-vector RRF retrieval followed by BGE reranking."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from elite_rag.config import get_settings
from elite_rag.multi import multi_settings
from elite_rag.rag import RagService, TriVectorRetriever


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            rows.append(value)
    return rows


def ranked_documents(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    seen: set[str] = set()
    for chunk in chunks:
        document_id = chunk.get("document_id")
        if not document_id or document_id in seen:
            continue
        seen.add(document_id)
        result.append({
            "document_id": document_id,
            "title": chunk.get("title"),
            "section_name": chunk.get("section_name"),
            "passage_id": chunk.get("passage_id"),
            "rank": len(result) + 1,
            "score": chunk.get("rerank_score", chunk.get("score")),
        })
    return result


def expected_ranks(documents: list[dict[str, Any]], expected: list[str]) -> dict[str, int]:
    ranks = {item["document_id"]: item["rank"] for item in documents}
    return {doc_id: ranks[doc_id] for doc_id in expected if doc_id in ranks}


def main() -> int:
    settings = multi_settings(get_settings())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(settings.questions_file))
    parser.add_argument("--output", type=Path, default=Path(settings.multi_retrieval_output))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    try:
        rows = read_jsonl(args.input)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    retriever = TriVectorRetriever(settings)
    service = RagService(settings, retriever=retriever)
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        evaluated = rows if args.limit is None else rows[: args.limit]
        manifest_path = args.output.with_suffix(".manifest.json")
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "collection": settings.qdrant_collection,
            "input": str(args.input),
            "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
            "output": str(args.output),
            "question_count": len(evaluated),
            "retrieval_number": settings.retrieval_number,
            "vectors": [
                settings.qdrant_dense_vector_name,
                settings.qdrant_sparse_vector_name,
                settings.qdrant_bm25_vector_name,
            ],
            "bm25_model": settings.bm25_model,
            "fusion": "rrf",
            "rerank_model": settings.rerank_model,
            "rerank_top_n": "all retrieved passages",
            "status": "running",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        failures = 0
        with args.output.open("w", encoding="utf-8") as handle:
            for index, row in enumerate(evaluated, start=1):
                question = str(row.get("question", row.get("query", ""))).strip()
                expected = row.get("expected_doc_ids", row.get("expected_doc_id", []))
                if isinstance(expected, str):
                    expected = [expected]
                if not isinstance(expected, list):
                    expected = []
                expected = [str(value) for value in expected if value]
                try:
                    candidates = retriever.search(question, settings.retrieval_number)
                    reranked = service.rerank(question, candidates) if candidates else []
                    qdrant_docs = ranked_documents(candidates)
                    reranked_docs = ranked_documents(reranked)
                    result = {
                        **row,
                        "collection": settings.qdrant_collection,
                        "retrieval_number": settings.retrieval_number,
                        "qdrant_candidate_count": len(candidates),
                        "reranker_candidate_count": len(candidates),
                        "expected_document_ranks": expected_ranks(qdrant_docs, expected),
                        "reranked_expected_document_ranks": expected_ranks(reranked_docs, expected),
                        "qdrant_documents": qdrant_docs,
                        "reranked_documents": reranked_docs,
                        "retrieval_status": "ok",
                    }
                    print(
                        f"{index}/{len(evaluated)} {row.get('question_id', index)} "
                        f"candidates={len(candidates)} expected_ranks="
                        f"{result['reranked_expected_document_ranks']}"
                    )
                except Exception as exc:
                    failures += 1
                    result = {**row, "retrieval_status": "error", "error": str(exc)}
                    print(f"{row.get('question_id', index)} failed: {exc}")
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
        manifest.update(
            status="complete" if failures == 0 else "completed_with_errors",
            failures=failures,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    finally:
        service.close()
    print(f"Results saved to {args.output}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
