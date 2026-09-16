from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

from elite_rag.config import Settings


def format_context_list(contexts: list[str]) -> str:
    return "\n\n".join(
        f"[Context Chunk {index}]\n{context.strip()}"
        for index, context in enumerate(contexts, start=1)
    )


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_recall: float | None = None
    context_precision: float | None = None
    answer_correctness: float | None = None


class LlamaEvaluator:
    """Text-based evaluator ported from rag-bench's Llama judge implementation."""

    def __init__(self, settings: Settings) -> None:
        self.base_url = settings.eval_base_url.rstrip("/")
        self.model = settings.eval_model
        self.api_key = settings.eval_api_key
        self.timeout = settings.eval_timeout_seconds

    def ask(self, prompt: str, temperature: float = 0.0) -> str:
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return str(response.json()["choices"][0]["message"]["content"]).strip()

    def evaluate(
        self,
        question: str,
        answer: str,
        ground_truth: str | None,
        contexts: list[str],
        metrics: set[str] | None = None,
    ) -> EvaluationResult:
        selected = metrics or {
            "faithfulness",
            "answer_relevancy",
            "context_recall",
            "context_precision",
            "answer_correctness",
        }
        requires_ground_truth = {"context_recall", "context_precision", "answer_correctness"}
        if selected & requires_ground_truth and not ground_truth:
            raise ValueError("ground_truth is required for the selected evaluation metrics")
        context = format_context_list(contexts)
        return EvaluationResult(
            faithfulness=(
                self._faithfulness(context, answer) if "faithfulness" in selected else None
            ),
            answer_relevancy=self._relevancy(question, answer)
            if "answer_relevancy" in selected
            else None,
            context_recall=self._context_recall(ground_truth, context)
            if "context_recall" in selected
            else None,
            context_precision=self._context_precision(question, ground_truth, context)
            if "context_precision" in selected
            else None,
            answer_correctness=self._correctness(answer, ground_truth)
            if "answer_correctness" in selected
            else None,
        )

    def _faithfulness(self, context: str, answer: str) -> float:
        claims = self._lines(self.ask(f"""
Analyze the following Answer and break it down into a bulleted list of independent,
single factual claims. Do not add or infer anything outside the text.

Answer: {answer}

List of claims:
"""))
        if not claims:
            return 0.0
        supported = sum(
            "YES" in self.ask(f"""
Determine if the following Claim can be logically inferred using ONLY the provided Context.
Respond with exactly one word: 'YES' if supported, or 'NO' otherwise.

Context: {context}
Claim: {claim}

Verdict (YES/NO):
""").upper()
            for claim in claims
        )
        return supported / len(claims)

    def _relevancy(self, question: str, answer: str) -> float:
        generated = self.ask(f"""
Generate the specific, concise question that the following answer is trying to resolve.
Only return the question text.

Answer: {answer}
Generated Question:
""")
        return self._float(self.ask(f"""
Compare these two questions. Rate their similarity from 0.0 to 1.0.
Respond with ONLY a numeric float value.

Original Question: {question}
Generated Question: {generated}

Similarity Score:
"""))

    def _context_recall(self, ground_truth: str | None, context: str) -> float:
        facts = self._lines(self.ask(f"""
Break down the following Ground Truth answer into a bulleted list of independent,
single factual statements. Only output the bulleted list.

Ground Truth: {ground_truth}

List of facts:
"""))
        if not facts:
            return 0.0
        found = sum(
            "YES" in self.ask(f"""
Can the following Fact be directly found or clearly deduced using ONLY the provided Context?
Respond with exactly one word: 'YES' if present, or 'NO' if missing.

Context: {context}
Fact: {fact}

Verdict (YES/NO):
""").upper()
            for fact in facts
        )
        return found / len(facts)

    def _context_precision(self, question: str, ground_truth: str | None, context: str) -> float:
        sentences = [part.strip() for part in context.replace("?", ".").split(".") if part.strip()]
        if not sentences:
            return 0.0
        relevant = sum(
            "YES" in self.ask(f"""
Is this context sentence highly relevant and useful for answering the question,
given the ground truth? Respond with exactly one word: 'YES' or 'NO'.

Question: {question}
Ground Truth Answer: {ground_truth}
Sentence: {sentence}

Verdict (YES/NO):
""").upper()
            for sentence in sentences
        )
        return relevant / len(sentences)

    def _correctness(self, answer: str, ground_truth: str | None) -> float:
        raw = self.ask(f"""
Compare the Generated Answer against the Ground Truth. Return strictly this JSON shape:
{{"TP": ["facts"], "FP": ["facts"], "FN": ["facts"]}}

Generated Answer: {answer}
Ground Truth: {ground_truth}

JSON Output:
""")
        try:
            data = json.loads(_strip_code_fence(raw))
            tp = len(data.get("TP", []))
            fp = len(data.get("FP", []))
            fn = len(data.get("FN", []))
            return tp / (tp + 0.5 * (fp + fn)) if tp + fp + fn else 0.0
        except (TypeError, ValueError, json.JSONDecodeError):
            return self._float(self.ask(f"""
Rate factual correctness from 0.0 to 1.0. Respond with ONLY the float.
Ground Truth: {ground_truth}
Generated Answer: {answer}
"""))

    @staticmethod
    def _lines(value: str) -> list[str]:
        return [
            re.sub(r"^\s*[-•]\s*", "", line).strip()
            for line in value.splitlines()
            if line.strip()
        ]

    @staticmethod
    def _float(value: str) -> float:
        try:
            return max(0.0, min(1.0, float(value.strip())))
        except ValueError:
            return 0.0


def _strip_code_fence(value: str) -> str:
    if "```json" in value:
        return value.split("```json", 1)[1].split("```", 1)[0].strip()
    if "```" in value:
        return value.split("```", 1)[1].split("```", 1)[0].strip()
    return value.strip()


def run_evaluation(
    evaluator: LlamaEvaluator,
    answers_path: Path,
    output_path: Path,
    *,
    limit: int | None = None,
) -> int:
    """Evaluate benchmark answer JSONL using the configured Llama judge."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    with answers_path.open(encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as output:
        for line in source:
            if not line.strip() or (limit is not None and processed >= limit):
                if limit is not None and processed >= limit:
                    break
                continue
            row = json.loads(line)
            result = evaluator.evaluate(
                str(row["question"]),
                str(row["answer"]),
                str(row.get("ground_truth", "")),
                [str(context) for context in row.get("contexts", [])],
            )
            output.write(json.dumps({**row, **asdict(result)}, ensure_ascii=False) + "\n")
            output.flush()
            processed += 1
            print(f"[{processed}] {row.get('question_id', 'unknown')}", flush=True)
    return processed
