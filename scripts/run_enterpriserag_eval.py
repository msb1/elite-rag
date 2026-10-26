#!/usr/bin/env python3
"""Run the EnterpriseRAG-Bench scorer through Elite RAG's local judge endpoint."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from openai import OpenAI

from elite_rag.config import Settings


def _chat_completions_generate(self: Any, messages: list[Any]):
    if self.tools:
        raise ValueError("The local evaluation adapter does not support tool calls")
    response = self.client.chat.completions.create(
        model=self.model,
        messages=[{"role": message.role, "content": message.content} for message in messages],
        stream=True,
    )
    for chunk in response:
        if not chunk.choices:
            continue
        content = chunk.choices[0].delta.content
        if isinstance(content, str) and content:
            yield content


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-repo", type=Path, default=Path("/private/tmp/EnterpriseRAG-Bench"))
    parser.add_argument("--answers-file", type=Path, default=Path("results/enterpriserag_answers.jsonl"))
    parser.add_argument("--questions-file", type=Path, default=Path("benchmark/questions.jsonl"))
    parser.add_argument("--results-file", type=Path, default=Path("results/enterpriserag_results.json"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    repo = args.bench_repo.resolve()
    if not (repo / "src/scripts/answer_evaluation/metrics_based_eval.py").is_file():
        parser.error(f"EnterpriseRAG-Bench checkout not found under {repo}")
    settings = Settings()
    base_url = settings.eval_base_url or settings.openai_local_endpoint
    os.environ["LLM_PROVIDER"] = "openai"
    os.environ["LLM_API_KEY"] = settings.eval_api_key
    os.environ["LLM_MODEL_NAME"] = settings.eval_model
    os.environ["OPENAI_BASE_URL"] = base_url
    os.environ["CHEAP_LLM_MODEL_NAME"] = settings.eval_model
    sys.path.insert(0, str(repo))

    from src.llm import openai_llm

    original_init = openai_llm.OpenAILLM.__init__

    def local_init(self: Any, *init_args: Any, **init_kwargs: Any) -> None:
        original_init(self, *init_args, **init_kwargs)
        self.client = OpenAI(
            api_key=settings.eval_api_key,
            base_url=base_url,
            timeout=300,
        )

    openai_llm.OpenAILLM.__init__ = local_init
    openai_llm.OpenAILLM.generate = _chat_completions_generate

    from src.scripts.answer_evaluation.metrics_based_eval import main as evaluate

    cli_args = [
        "--answers-file", str(args.answers_file.resolve()),
        "--questions-file", str(args.questions_file.resolve()),
        "--results-file", str(args.results_file.resolve()),
        "--no-correction",
        "--skip-citation-stripping",
        "--parallelism", str(args.parallelism),
    ]
    if args.limit is not None:
        cli_args.extend(("--limit", str(args.limit)))
    if args.resume:
        cli_args.append("--resume")
    sys.argv = ["metrics_based_eval", *cli_args]
    print(f"EnterpriseRAG-Bench eval via {base_url}; model={settings.eval_model}")
    evaluate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
