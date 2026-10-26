"""Exact expected-parent-document ID coverage of retrieved contexts."""

from __future__ import annotations

import statistics
from typing import Any


def document_retrieval(record: dict[str, Any]) -> dict[str, Any]:
    expected = list(dict.fromkeys(record.get("expected_doc_ids") or []))
    if not expected:
        return {"status": "no_expected_documents"}
    sources = record.get("sources")
    if not isinstance(sources, list) or any(
        not isinstance(source, dict) or not source.get("document_id") for source in sources
    ):
        return {"status": "missing_document_ids"}
    # Score unique parent documents in first-hit order. Duplicate child hits
    # from one parent never consume another document rank.
    ids = [
        str(source.get("parent_document_id") or source["document_id"])
        for source in sources
    ]
    retrieved = list(dict.fromkeys(ids))
    expected_set = set(expected)

    def scores(selected: list[str]) -> dict[str, float]:
        matched = expected_set & set(selected)
        first = next((i for i, doc in enumerate(selected, 1) if doc in expected_set), None)
        return {
            "document_recall": len(matched) / len(expected),
            "document_precision": len(matched) / len(selected) if selected else 0.0,
            "target_doc_hit_rate": float(bool(matched)),
            "all_expected_found": float(expected_set <= set(selected)),
            "reciprocal_rank": 1 / first if first else 0.0,
        }

    return {
        "status": "ok",
        "expected_doc_ids": expected,
        "retrieved_doc_ids": retrieved,
        "matched_doc_ids": [doc for doc in expected if doc in retrieved],
        "missing_doc_ids": [doc for doc in expected if doc not in retrieved],
        "first_relevant_document_rank": next(
            (i for i, doc in enumerate(retrieved, 1) if doc in expected_set), None
        ),
        "source_context_count": len(sources),
        "unique_document_count": len(retrieved),
        "expected_count_exceeds_retrieved_document_count": len(expected) > len(retrieved),
        **scores(retrieved),
        # @k means "up to k ranked parents"; a short or empty result still counts
        # as a miss rather than disappearing from the aggregate denominator.
        "at_k": {
            str(k): scores(retrieved[:k])
            for k in (1, 3, 5)
        },
    }


def summarize_document_retrieval(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if row["document_retrieval"]["status"] == "ok"]

    def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
        names = (
            "document_recall",
            "document_precision",
            "target_doc_hit_rate",
            "all_expected_found",
            "reciprocal_rank",
        )
        return {
            "count": len(items),
            **{
                name: statistics.fmean(item[name] for item in items) if items else None
                for name in names
            },
        }

    details = [row["document_retrieval"] for row in scored]
    summary = aggregate(details)
    summary.update(
        {
            "no_expected_documents": sum(
                row["document_retrieval"]["status"] == "no_expected_documents" for row in rows
            ),
            "missing_document_ids": sum(
                row["document_retrieval"]["status"] == "missing_document_ids" for row in rows
            ),
            "no_expected_found": sum(not item["matched_doc_ids"] for item in details),
            "some_expected_missing": sum(bool(item["missing_doc_ids"]) for item in details),
            "expected_count_exceeds_retrieved_document_count": sum(
                item["expected_count_exceeds_retrieved_document_count"] for item in details
            ),
            "at_k": {
                str(k): aggregate(
                    [item["at_k"][str(k)] for item in details]
                )
                for k in (1, 3, 5)
            },
        }
    )
    for field in ("source_types", "question_type"):
        groups = sorted(
            {
                value
                for row in scored
                for value in (
                    row.get(field, []) if field == "source_types" else [row.get(field, "unknown")]
                )
            }
        )
        summary["by_" + field] = {
            group: aggregate(
                [
                    row["document_retrieval"]
                    for row in scored
                    if (
                        group in row.get(field, [])
                        if field == "source_types"
                        else row.get(field, "unknown") == group
                    )
                ]
            )
            for group in groups
        }
    summary["judge_recall_comparison"] = {}
    for label, hit in (("expected_document_found", True), ("no_expected_document_found", False)):
        judged = [
            row
            for row in scored
            if bool(row["document_retrieval"]["target_doc_hit_rate"]) == hit
            and row.get("evaluation_status") == "ok"
            and isinstance(row.get("context_recall"), (int, float))
        ]
        summary["judge_recall_comparison"][label] = {
            "count": len(judged),
            "mean_context_recall": statistics.fmean(row["context_recall"] for row in judged)
            if judged
            else None,
            "context_recall_below_0_30": sum(row["context_recall"] < 0.30 for row in judged),
        }
    return summary
