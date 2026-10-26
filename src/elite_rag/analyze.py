"""Summarize metrics and export severe faithfulness and retrieval failures."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from elite_rag.config import Settings
from elite_rag.retrieval_metrics import document_retrieval, summarize_document_retrieval

METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_recall",
    "context_precision",
    "answer_correctness",
)


def _read(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    ordered_keys: list[tuple[str, str]] = []
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for index, row in enumerate(rows):
        identifier = row.get("question_id")
        key = ("question", str(identifier)) if identifier is not None else ("line", str(index))
        if key not in latest:
            ordered_keys.append(key)
        latest[key] = row
    return [latest[key] for key in ordered_keys]


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_analysis(settings: Settings, *, input_path: str | None = None) -> dict[str, Any]:
    source = Path(input_path or settings.eval_output_file)
    all_rows = [dict(row, document_retrieval=document_retrieval(row)) for row in _read(source)]
    rows = [row for row in all_rows if row.get("evaluation_status") == "ok"]
    summary: dict[str, Any] = {
        "records": len(rows),
        "input_records": len(all_rows),
        "metrics": {},
        "document_retrieval": summarize_document_retrieval(all_rows),
    }
    for metric in METRICS:
        values = [row[metric] for row in rows if isinstance(row.get(metric), (int, float))]
        summary["metrics"][metric] = {
            "count": len(values),
            "mean": statistics.fmean(values) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "stddev": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
        }
    hallucinations = [row for row in rows if row.get("faithfulness", 1) < 0.30]
    misses = [row for row in rows if row.get("context_recall", 1) < 0.30]
    _write(Path(settings.hallucinations_output_file), hallucinations)
    _write(Path(settings.retrieval_misses_output_file), misses)
    summary["hallucination_failures"] = len(hallucinations)
    summary["retrieval_misses"] = len(misses)
    output = Path(settings.analysis_output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    retrieval_output = output.with_name(output.stem + "_document_retrieval.jsonl")
    _write(
        retrieval_output,
        [
            {
                key: row[key]
                for key in (
                    "question_id",
                    "question",
                    "question_type",
                    "source_types",
                    "evaluation_status",
                    "context_recall",
                    "context_precision",
                    "document_retrieval",
                )
                if key in row
            }
            for row in all_rows
        ],
    )
    summary["document_retrieval_output"] = str(retrieval_output)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_dashboard(Path(settings.dashboard_output_file), summary["metrics"])
    for metric, stats in summary["metrics"].items():
        mean = "n/a" if stats["mean"] is None else f"{stats['mean']:.3f}"
        print(f"{metric}: mean={mean} (n={stats['count']})")
    print(f"Faithfulness < 0.30: {len(hallucinations)}")
    print(f"Context recall < 0.30: {len(misses)}")
    retrieval = summary["document_retrieval"]
    print(f"Document retrieval (n={retrieval['count']}, independent of judge status):")
    for name in (
        "target_doc_hit_rate",
        "document_recall",
        "document_precision",
        "all_expected_found",
        "reciprocal_rank",
    ):
        value = retrieval[name]
        print(f"  {name}: {value:.3f}" if value is not None else f"  {name}: n/a")
    hit_at_five = retrieval["at_k"]["5"]
    hit_at_five_value = hit_at_five["target_doc_hit_rate"]
    hit_at_five_mean = (
        f"{hit_at_five_value:.3f}" if hit_at_five_value is not None else "n/a"
    )
    print(f"  target_doc_hit_rate@5: {hit_at_five_mean} (n={hit_at_five['count']})")
    print(f"  No expected document found: {retrieval['no_expected_found']}")
    print(f"  No expected documents (excluded): {retrieval['no_expected_documents']}")
    print(f"Document retrieval details: {retrieval_output}")
    print(f"Analysis: {output}")
    return summary


def _write_dashboard(path: Path, metrics: dict[str, dict[str, Any]]) -> None:
    width, row_height, left, right = 900, 54, 270, 40
    height = 100 + row_height * len(METRICS)
    bar_width = width - left - right
    items = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="30" y="42" font-family="sans-serif" font-size="24" '
        'font-weight="bold" fill="#172554">RAG Evaluation Metrics</text>',
    ]
    for index, name in enumerate(METRICS):
        y = 78 + index * row_height
        value = metrics[name]["mean"]
        items.append(
            f'<text x="30" y="{y + 23}" font-family="sans-serif" font-size="15" '
            f'fill="#1f2937">{escape(name.replace("_", " ").title())}</text>'
        )
        items.append(
            f'<rect x="{left}" y="{y}" width="{bar_width}" height="30" rx="5" fill="#e5e7eb"/>'
        )
        if value is not None:
            items.append(
                f'<rect x="{left}" y="{y}" width="{bar_width * value:.2f}" '
                'height="30" rx="5" fill="#2563eb"/>'
            )
            label = f"{value:.3f}"
        else:
            label = "n/a"
        items.append(
            f'<text x="{left + bar_width + 8}" y="{y + 21}" font-family="sans-serif" '
            f'font-size="14" fill="#1f2937">{label}</text>'
        )
    items.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(items) + "\n", encoding="utf-8")
