"""Custom RAGAS-style text judge metrics ported from the Python RAG benchmark."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from openai import OpenAI

from elite_rag.config import Settings


def format_context_list(contexts: Sequence[str]) -> str:
    return "\n\n".join(
        f"[Context Chunk {index}]\n{context.strip()}"
        for index, context in enumerate(contexts, start=1)
    )


class RagEvaluator:
    def __init__(self, settings: Settings, client: OpenAI | None = None) -> None:
        self.settings = settings
        self.client = client or OpenAI(
            base_url=settings.eval_base_url or settings.openai_local_endpoint,
            api_key=settings.eval_api_key,
            timeout=180,
        )

    def ask(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.settings.eval_model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return (response.choices[0].message.content or "").strip()

    def _claims(self, answer: str) -> list[str]:
        text = self.ask(
            "Analyze the answer and list its independent factual claims, one per line. "
            "Do not infer or add claims. Return only the list.\n\nAnswer:\n" + answer
        )
        return [
            re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
            for line in text.splitlines()
            if line.strip()
        ]

    def _facts(self, ground_truth: str) -> list[str]:
        text = self.ask(
            "Break the ground truth into independent factual statements, one per line. "
            "Return only the list.\n\nGround truth:\n" + ground_truth
        )
        return [
            re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
            for line in text.splitlines()
            if line.strip()
        ]

    @staticmethod
    def _yes(value: str) -> bool:
        normalized = value.strip().upper()
        return normalized == "YES" or normalized.startswith("YES.")

    def faithfulness(self, context: str, answer: str) -> float:
        claims = self._claims(answer)
        if not claims:
            return 0.0
        supported = sum(
            self._yes(self.ask(
                "Can this claim be supported using only the provided context? "
                "Respond exactly YES or NO.\n\n"
                f"Context:\n{context}\n\nClaim:\n{claim}"
            ))
            for claim in claims
        )
        return supported / len(claims)

    def answer_relevancy(self, question: str, answer: str) -> float:
        generated = self.ask(
            "Generate the specific concise question this answer resolves. "
            "Return only the question.\n\nAnswer:\n" + answer
        )
        result = self.ask(
            "Compare the semantic intent of the two questions. Return only a number from "
            "0.0 to 1.0.\n\n"
            f"Original question: {question}\nGenerated question: {generated}"
        )
        return self._score(result)

    def answer_correctness(self, answer: str, ground_truth: str) -> float:
        prompt = (
            "Compare the generated answer with the ground truth. Return strict JSON with "
            "arrays TP (facts in both), FP (facts in answer but unsupported or conflicting), "
            "and FN (facts in ground truth missed by the answer).\n\n"
            f"Generated answer:\n{answer}\n\nGround truth:\n{ground_truth}"
        )
        raw = self.ask(prompt)
        try:
            start, end = raw.find("{"), raw.rfind("}")
            data = json.loads(raw[start : end + 1])
            tp, fp, fn = (len(data.get(key, [])) for key in ("TP", "FP", "FN"))
            denominator = tp + 0.5 * (fp + fn)
            return tp / denominator if denominator else 0.0
        except (ValueError, TypeError, json.JSONDecodeError):
            score = self._score(self.ask(
                "Rate factual correctness against the ground truth from 0 to 1. "
                "Return only a number.\n\n"
                f"Ground truth:\n{ground_truth}\n\nGenerated answer:\n{answer}"
            ))
            return score

    def context_recall(self, ground_truth: str, context: str) -> float:
        facts = self._facts(ground_truth)
        if not facts:
            return 0.0
        found = sum(
            self._yes(self.ask(
                "Can this fact be found or clearly deduced using only the provided context? "
                "Respond exactly YES or NO.\n\n"
                f"Context:\n{context}\n\nFact:\n{fact}"
            ))
            for fact in facts
        )
        return found / len(facts)

    def context_precision(self, question: str, ground_truth: str, context: str) -> float:
        sentences = [
            part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", context) if part.strip()
        ]
        if not sentences:
            return 0.0
        relevant = sum(
            self._yes(self.ask(
                "Is this context sentence useful for answering the question, given the "
                "ground truth? Respond exactly YES or NO.\n\n"
                f"Question: {question}\nGround truth: {ground_truth}\n"
                f"Sentence: {sentence}"
            ))
            for sentence in sentences
        )
        return relevant / len(sentences)

    @staticmethod
    def _score(value: str) -> float:
        match = re.search(r"[-+]?(?:\d+\.?\d*|\.\d+)", value)
        if not match:
            return 0.0
        score = float(match.group())
        return min(1.0, max(0.0, score))

    def evaluate(self, record: dict[str, Any]) -> dict[str, float]:
        contexts = record.get("contexts", [])
        if isinstance(contexts, str):
            contexts = [contexts]
        if not isinstance(contexts, list):
            contexts = []
        context = format_context_list([str(item) for item in contexts])
        question = str(record.get("question", ""))
        answer = str(record.get("answer", ""))
        ground_truth = str(record.get("ground_truth", ""))
        return {
            "faithfulness": self.faithfulness(context, answer),
            "answer_relevancy": self.answer_relevancy(question, answer),
            "context_recall": self.context_recall(ground_truth, context),
            "context_precision": self.context_precision(question, ground_truth, context),
            "answer_correctness": self.answer_correctness(answer, ground_truth),
        }
