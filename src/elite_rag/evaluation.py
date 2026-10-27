"""Use RAGAS 0.4.3 prompts and scoring with the configured text endpoints."""

from __future__ import annotations

import math
import json
from collections.abc import Sequence
from typing import Any, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError
from ragas.metrics.collections.answer_correctness.util import (
    ClassificationWithReason,
    CorrectnessClassifierInput,
    CorrectnessClassifierPrompt,
)
from ragas.metrics.collections.answer_correctness.util import (
    StatementGeneratorInput as CorrectnessStatementInput,
)
from ragas.metrics.collections.answer_correctness.util import (
    StatementGeneratorOutput as CorrectnessStatementOutput,
)
from ragas.metrics.collections.answer_correctness.util import (
    StatementGeneratorPrompt as CorrectnessStatementPrompt,
)
from ragas.metrics.collections.answer_relevancy.util import (
    AnswerRelevanceInput,
    AnswerRelevanceOutput,
    AnswerRelevancePrompt,
)
from ragas.metrics.collections.context_precision.util import (
    ContextPrecisionInput,
    ContextPrecisionOutput,
    ContextPrecisionPrompt,
)
from ragas.metrics.collections.context_recall.util import (
    ContextRecallInput,
    ContextRecallOutput,
    ContextRecallPrompt,
)
from ragas.metrics.collections.faithfulness.util import (
    NLIStatementInput,
    NLIStatementOutput,
    NLIStatementPrompt,
    StatementGeneratorInput,
    StatementGeneratorOutput,
    StatementGeneratorPrompt,
)

from elite_rag.config import Settings
from elite_rag.embedding import HybridEmbedder

EVALUATION_IMPLEMENTATION = "ragas-0.4.3-prompts-json-parent-context-v4-structured-output"
JUDGE_SYSTEM_PROMPT = (
    "Follow the user's evaluation instructions exactly. Return only one valid JSON object "
    "matching the requested response schema. Populate the schema's fields with the result; "
    "do not return the schema definition, markdown, or any extra text."
)
Output = TypeVar("Output", bound=BaseModel)


class EvaluationError(ValueError):
    """An evaluation failure annotated with its metric stage and judge output."""

    def __init__(
        self,
        stage: str,
        message: str,
        raw_responses: Sequence[str] = (),
    ) -> None:
        super().__init__(f"{stage}: {message}")
        self.stage = stage
        self.raw_responses = list(raw_responses)


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
        self.embedder = HybridEmbedder(
            settings.embedding_model,
            settings.embedding_dimension,
            settings.embedding_base_url,
            settings.embedding_api_key,
        )
        self.prompt_retries: list[dict[str, Any]] = []

    @staticmethod
    def _strict_json_schema(output: type[Output]) -> dict[str, Any]:
        """Return an OpenAI Structured Outputs compatible schema for a Pydantic model."""
        schema = output.model_json_schema()

        def make_strict(node: Any) -> None:
            if isinstance(node, dict):
                if node.get("type") == "object":
                    node["additionalProperties"] = False
                for value in node.values():
                    make_strict(value)
            elif isinstance(node, list):
                for value in node:
                    make_strict(value)

        make_strict(schema)
        return {
            "type": "json_schema",
            "json_schema": {
                "name": output.__name__,
                "strict": True,
                "schema": schema,
            },
        }

    def ask(self, prompt: str, output: type[Output]) -> str:
        response = self.client.chat.completions.create(
            model=self.settings.eval_model,
            temperature=0,
            response_format=self._strict_json_schema(output),
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return (response.choices[0].message.content or "").strip()

    @staticmethod
    def _parse_output(raw: str, output: type[Output]) -> Output:
        """Find and validate a JSON object, allowing surrounding prose or code fences."""
        decoder = json.JSONDecoder()
        errors: list[str] = []
        for index, character in enumerate(raw):
            if character != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(raw, index)
            except json.JSONDecodeError:
                continue
            try:
                return output.model_validate(candidate)
            except ValidationError as exc:
                errors.append(str(exc))
        if errors:
            raise ValueError("JSON objects did not match the expected output model: " + errors[-1])
        raise ValueError("response contained no valid JSON object")

    def _prompt(
        self,
        prompt: Any,
        data: BaseModel,
        output: type[Output],
        *,
        stage: str,
    ) -> Output:
        prompt_text = prompt.to_string(data)
        raw_responses: list[str] = []
        try:
            raw = self.ask(prompt_text, output)
        except Exception as exc:
            raise EvaluationError(
                stage,
                f"judge request failed with strict JSON Schema {output.__name__}: {exc}",
            ) from exc
        raw_responses.append(raw)
        try:
            return self._parse_output(raw, output)
        except (ValueError, ValidationError) as first_error:
            first_error_text = str(first_error)
            self.prompt_retries.append(
                {
                    "stage": stage,
                    "attempt": 1,
                    "validation_error": first_error_text,
                    "raw_response": raw,
                }
            )
            correction = (
                "Your previous response did not contain a valid instance of the required "
                "JSON output. Return only the populated JSON object matching the schema "
                "already provided above. Do not repeat the schema definition. "
                f"Validation issue: {first_error_text[:1000]}"
            )

        try:
            retry_raw = self.ask(f"{prompt_text}\n\n{correction}", output)
        except Exception as exc:
            raise EvaluationError(
                stage,
                f"judge retry failed with strict JSON Schema {output.__name__}: {exc}",
                raw_responses,
            ) from exc
        raw_responses.append(retry_raw)
        try:
            return self._parse_output(retry_raw, output)
        except (ValueError, ValidationError) as exc:
            raise EvaluationError(
                stage,
                f"invalid {output.__name__} response after one correction retry: {exc}",
                raw_responses,
            ) from exc

    def faithfulness(self, question: str, answer: str, contexts: list[str]) -> float:
        statements = self._prompt(
            StatementGeneratorPrompt(),
            StatementGeneratorInput(question=question, answer=answer),
            StatementGeneratorOutput,
            stage="faithfulness.statement_generation",
        ).statements
        if not statements:
            return math.nan
        verdicts = self._prompt(
            NLIStatementPrompt(),
            NLIStatementInput(context="\n".join(contexts), statements=statements),
            NLIStatementOutput,
            stage="faithfulness.verdicts",
        ).statements
        return sum(bool(item.verdict) for item in verdicts) / len(verdicts)

    def answer_relevancy(self, question: str, answer: str) -> float:
        generated = [
            self._prompt(
                AnswerRelevancePrompt(),
                AnswerRelevanceInput(response=answer),
                AnswerRelevanceOutput,
                stage="answer_relevancy.question_generation",
            )
            for _ in range(3)
        ]
        questions = [item.question for item in generated if item.question]
        if not questions:
            return 0.0
        try:
            vectors = [item.dense for item in self.embedder.embed_queries([question, *questions])]
        except Exception as exc:
            raise EvaluationError("answer_relevancy.embeddings", str(exc)) from exc
        original, candidates = vectors[0], vectors[1:]
        norm_original = math.sqrt(sum(a * a for a in original))
        similarities = []
        for candidate in candidates:
            norm_candidate = math.sqrt(sum(b * b for b in candidate))
            similarities.append(
                sum(a * b for a, b in zip(original, candidate, strict=True))
                / (norm_original * norm_candidate)
            )
        score = sum(similarities) / len(similarities)
        noncommittal = [item.noncommittal for item in generated if item.question]
        return score * int(not all(noncommittal))

    def context_recall(self, question: str, reference: str, contexts: list[str]) -> float:
        result = self._prompt(
            ContextRecallPrompt(),
            ContextRecallInput(question=question, context="\n".join(contexts), answer=reference),
            ContextRecallOutput,
            stage="context_recall.classification",
        )
        verdicts = result.classifications
        return (
            sum(item.attributed for item in verdicts) / len(verdicts)
            if verdicts
            else math.nan
        )

    def context_precision(self, question: str, reference: str, contexts: list[str]) -> float:
        prompt = ContextPrecisionPrompt()
        verdicts = []
        for index, context in enumerate(contexts, start=1):
            result = self._prompt(
                prompt,
                ContextPrecisionInput(question=question, context=context, answer=reference),
                ContextPrecisionOutput,
                stage=f"context_precision.context_{index}.verdict",
            )
            verdicts.append(result.verdict)
        relevant = 0
        numerator = 0.0
        for rank, verdict in enumerate(verdicts, 1):
            relevant += verdict
            numerator += (relevant / rank) * verdict
        return numerator / (sum(verdicts) + 1e-10)

    def answer_correctness(self, question: str, answer: str, reference: str) -> float:
        prompt = CorrectnessStatementPrompt()
        answer_statements = self._prompt(
            prompt,
            CorrectnessStatementInput(question=question, answer=answer),
            CorrectnessStatementOutput,
            stage="answer_correctness.answer_statement_generation",
        ).statements
        reference_statements = self._prompt(
            prompt,
            CorrectnessStatementInput(question=question, answer=reference),
            CorrectnessStatementOutput,
            stage="answer_correctness.reference_statement_generation",
        ).statements
        if answer_statements and reference_statements:
            classified = self._prompt(
                CorrectnessClassifierPrompt(),
                CorrectnessClassifierInput(
                    question=question,
                    answer=answer_statements,
                    ground_truth=reference_statements,
                ),
                ClassificationWithReason,
                stage="answer_correctness.statement_classification",
            )
            tp, fp, fn = len(classified.TP), len(classified.FP), len(classified.FN)
            precision = tp / (tp + fp) if tp + fp else (1.0 if not fn else 0.0)
            recall = tp / (tp + fn) if tp + fn else (1.0 if not fp else 0.0)
            factuality = (
                2 * precision * recall / (precision + recall) if precision + recall else 0.0
            )
        else:
            factuality = 1.0
        try:
            vectors = [item.dense for item in self.embedder.embed_documents([answer, reference])]
        except Exception as exc:
            raise EvaluationError("answer_correctness.embeddings", str(exc)) from exc
        left, right = vectors
        norm_left = math.sqrt(sum(a * a for a in left))
        norm_right = math.sqrt(sum(b * b for b in right))
        similarity = (
            sum(a * b for a, b in zip(left, right, strict=True)) / (norm_left * norm_right)
            if norm_left and norm_right
            else 0.0
        )
        return 0.75 * factuality + 0.25 * similarity

    def evaluate(self, record: dict[str, Any]) -> dict[str, float]:
        self.prompt_retries = []
        question = record.get("question")
        answer = record.get("answer")
        reference = record.get("ground_truth")
        contexts = record.get("contexts")
        missing = [
            name
            for name, value in (
                ("question", question),
                ("answer", answer),
                ("ground_truth", reference),
            )
            if not isinstance(value, str) or not value
        ]
        if missing:
            raise EvaluationError(
                "input_validation",
                "required non-empty string field(s) missing or invalid: " + ", ".join(missing),
            )
        if not isinstance(contexts, list) or not contexts:
            raise EvaluationError(
                "input_validation", "contexts must be a non-empty list of strings"
            )
        if any(not isinstance(context, str) for context in contexts):
            raise EvaluationError(
                "input_validation", "contexts must contain only strings"
            )
        metrics = {}

        def add(name: str, score: float) -> None:
            if not math.isfinite(score):
                raise ValueError(f"RAGAS {name} returned a nonfinite score")
            metrics[name] = score

        stages = (
            ("faithfulness", lambda: self.faithfulness(question, answer, contexts)),
            ("answer_relevancy", lambda: self.answer_relevancy(question, answer)),
            ("context_recall", lambda: self.context_recall(question, reference, contexts)),
            ("context_precision", lambda: self.context_precision(question, reference, contexts)),
            ("answer_correctness", lambda: self.answer_correctness(question, answer, reference)),
        )
        for name, calculate in stages:
            try:
                add(name, calculate())
            except EvaluationError:
                raise
            except Exception as exc:
                raise EvaluationError(name, str(exc)) from exc
        return metrics
