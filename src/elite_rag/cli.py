"""Command line entry points for the standalone Elite RAG application."""

from __future__ import annotations

import argparse
import json

from elite_rag.analyze import run_analysis
from elite_rag.benchmark import run_benchmark
from elite_rag.config import get_settings
from elite_rag.eval import run_evaluation
from elite_rag.rag import RagService


def serve() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run("elite_rag.api:app", host=settings.server_host, port=settings.server_port)


def main() -> int:
    parser = argparse.ArgumentParser(prog="elite-rag")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("serve", help="Start the HTTP API server")
    rag_parser = subparsers.add_parser("rag", help="Answer one question")
    rag_parser.add_argument("question")
    benchmark_parser = subparsers.add_parser("benchmark", help="Run the bundled RAG benchmark")
    benchmark_parser.add_argument("--limit", type=int)
    benchmark_parser.add_argument("--question-id")
    benchmark_parser.add_argument("--output")
    eval_parser = subparsers.add_parser("eval", help="Evaluate generated benchmark answers")
    eval_parser.add_argument("--input")
    eval_parser.add_argument("--output")
    eval_parser.add_argument("--limit", type=int)
    analyze_parser = subparsers.add_parser("analyze", help="Summarize evaluation output")
    analyze_parser.add_argument("--input")
    args = parser.parse_args()
    settings = get_settings()
    if args.command == "serve":
        serve()
        return 0
    if args.command == "rag":
        service = RagService(settings)
        try:
            print(json.dumps(service.answer(args.question), ensure_ascii=False, indent=2))
        finally:
            service.close()
        return 0
    if args.command == "benchmark":
        return run_benchmark(
            settings, limit=args.limit, question_id=args.question_id, output_path=args.output
        )
    if args.command == "eval":
        return run_evaluation(
            settings, input_path=args.input, output_path=args.output, limit=args.limit
        )
    if args.command == "analyze":
        run_analysis(settings, input_path=args.input)
        return 0
    return 2
