#!/usr/bin/env python3
"""Run one grounded RAG query or the Enterprise RAG question benchmark."""

from __future__ import annotations

import argparse
import json

from elite_rag.benchmark import run_benchmark
from elite_rag.config import get_settings
from elite_rag.rag import RagService


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", help="Answer one question instead of running the benchmark")
    parser.add_argument("--benchmark", action="store_true", help="Run the benchmark dataset")
    parser.add_argument("--limit", type=int, help="Maximum new benchmark rows to generate")
    parser.add_argument("--question-id", help="Run only this benchmark question ID")
    parser.add_argument("--output", help="Override benchmark JSONL output")
    args = parser.parse_args()
    settings = get_settings()
    if args.question:
        service = RagService(settings)
        try:
            result = service.answer(args.question)
        finally:
            service.close()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.benchmark or args.question_id or args.limit:
        return run_benchmark(
            settings,
            limit=args.limit,
            question_id=args.question_id,
            output_path=args.output,
        )
    parser.error("provide --question or --benchmark")


if __name__ == "__main__":
    raise SystemExit(main())
