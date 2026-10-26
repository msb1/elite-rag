import json
from collections import deque
from io import BytesIO

import httpx
import pytest
from openai import OpenAI
from ragas.metrics.collections.answer_correctness.util import (
    CorrectnessClassifierPrompt,
)
from ragas.metrics.collections.answer_correctness.util import (
    StatementGeneratorPrompt as CorrectnessStatementPrompt,
)
from ragas.metrics.collections.answer_relevancy.util import AnswerRelevancePrompt
from ragas.metrics.collections.context_precision.util import ContextPrecisionPrompt
from ragas.metrics.collections.context_recall.util import ContextRecallPrompt
from ragas.metrics.collections.faithfulness.util import (
    NLIStatementPrompt,
)
from ragas.metrics.collections.faithfulness.util import (
    StatementGeneratorPrompt as FaithfulnessStatementPrompt,
)

from elite_rag.config import Settings
from elite_rag.eval import run_evaluation
from elite_rag.evaluation import RagEvaluator


@pytest.fixture
def evaluation_backend(monkeypatch):
    monkeypatch.setenv("RAGAS_DO_NOT_TRACK", "true")
    statements = ["The limit is ten.", "The limit is twenty."]
    outputs = deque(
        [
            {"statements": statements},
            {
                "statements": [
                    {"statement": text, "reason": "Fixture evidence.", "verdict": verdict}
                    for text, verdict in zip(statements, [1, 0], strict=True)
                ]
            },
            *[{"question": f"Generated question {i}", "noncommittal": 0} for i in range(3)],
            {
                "classifications": [
                    {"statement": text, "reason": "Fixture evidence.", "attributed": verdict}
                    for text, verdict in zip(statements, [1, 0], strict=True)
                ]
            },
            *[{"reason": "Fixture evidence.", "verdict": verdict} for verdict in [1, 0, 1]],
            {"statements": statements},
            {"statements": [statements[0], "The limit is thirty."]},
            {
                key: [{"statement": text, "reason": "Fixture evidence."}]
                for key, text in zip(
                    ("TP", "FP", "FN"), (*statements, "The limit is thirty."), strict=True
                )
            },
        ]
    )
    vectors = {
        "Original question": [1.0, 0.0],
        "Generated question 0": [1.0, 0.0],
        "Generated question 1": [0.0, 1.0],
        "Generated question 2": [0.6, 0.8],
        "Generated answer": [1.0, 0.0],
        "Reference answer": [0.8, 0.6],
    }
    chat_requests = []
    embedding_requests = []

    def respond(request):
        body = json.loads(request.content)
        chat_requests.append(body)
        return httpx.Response(
            200,
            json={
                "id": "fixture",
                "object": "chat.completion",
                "created": 0,
                "model": "fixture-judge",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(outputs.popleft()),
                        },
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    def embed(request, timeout):
        body = json.loads(request.data)
        texts = body["input"]
        embedding_requests.append((request.full_url, body))
        return BytesIO(
            json.dumps(
                {
                    "data": [
                        {
                            "index": i,
                            "embedding": vectors[text],
                            "sparse_vector": {"indices": [], "values": []},
                        }
                        for i, text in enumerate(texts)
                    ]
                }
            ).encode()
        )

    monkeypatch.setattr("elite_rag.embedding.urlopen", embed)
    client = OpenAI(
        base_url="http://judge.test/v1",
        api_key="fixture",
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    )
    settings = Settings(
        _env_file=None,
        eval_base_url="http://judge.test/v1",
        eval_model="fixture-judge",
        embedding_base_url="http://embedding.test/v1",
        embedding_model="bge-m3",
        embedding_dimension=2,
    )
    return RagEvaluator(settings, client), chat_requests, embedding_requests, outputs


SAMPLE = {
    "question": "Original question",
    "answer": "Generated answer",
    "ground_truth": "Reference answer",
    "contexts": ["PASSAGE_A\nFirst sentence. Second sentence.", "PASSAGE_B", "PASSAGE_C"],
}


@pytest.mark.parametrize("noncommittal", [0, 1])
def test_ragas_prompts_and_scores_use_plain_chat_completions(evaluation_backend, noncommittal):
    evaluator, chat_requests, embedding_requests, outputs = evaluation_backend
    for output in outputs:
        if "noncommittal" in output:
            output["noncommittal"] = noncommittal

    scores = evaluator.evaluate(SAMPLE)

    assert scores == pytest.approx(
        {
            "faithfulness": 0.5,
            "answer_relevancy": ((1 + 0 + 0.6) / 3) * (1 - noncommittal),
            "context_recall": 0.5,
            "context_precision": (1 + 2 / 3) / (2 + 1e-10),
            "answer_correctness": 0.75 * 0.5 + 0.25 * 0.8,
        }
    )
    assert not outputs
    assert len(chat_requests) == 12
    assert all("response_format" not in request for request in chat_requests)
    assert all(request["model"] == "fixture-judge" for request in chat_requests)
    prompts = [request["messages"][-1]["content"] for request in chat_requests]
    for prompt, instruction in zip(
        prompts,
        [
            FaithfulnessStatementPrompt.instruction,
            NLIStatementPrompt.instruction,
            *[AnswerRelevancePrompt.instruction] * 3,
            ContextRecallPrompt.instruction,
            *[ContextPrecisionPrompt.instruction] * 3,
            *[CorrectnessStatementPrompt.instruction] * 2,
            CorrectnessClassifierPrompt.instruction,
        ],
        strict=True,
    ):
        assert instruction in prompt
    for prompt, context in zip(prompts[6:9], SAMPLE["contexts"], strict=True):
        assert context.replace("\n", "\\n") in prompt
        assert sum(label in prompt for label in ("PASSAGE_A", "PASSAGE_B", "PASSAGE_C")) == 1
    assert len(embedding_requests) == 2
    assert embedding_requests[0][0] == "http://embedding.test/v1/embeddings"
    assert embedding_requests[0][1]["input"] == [
        "Original question",
        "Generated question 0",
        "Generated question 1",
        "Generated question 2",
    ]
    assert embedding_requests[1][1]["input"] == ["Generated answer", "Reference answer"]


def test_nonfinite_faithfulness_is_an_error(evaluation_backend):
    evaluator, chat_requests, _, outputs = evaluation_backend
    outputs.clear()
    outputs.append({"statements": []})
    with pytest.raises(ValueError, match="RAGAS faithfulness returned a nonfinite score"):
        evaluator.evaluate(SAMPLE)
    assert len(chat_requests) == 1


def test_batch_rejects_previous_partial_ragas_implementation(tmp_path):
    source = tmp_path / "input.jsonl"
    destination = tmp_path / "output.jsonl"
    source.write_text(json.dumps(SAMPLE) + "\n")
    previous = (
        json.dumps(
            {
                "question_id": "q1",
                "evaluation_status": "ok",
                "context_metrics_implementation": "ragas-0.4.3",
            }
        )
        + "\n"
    )
    destination.write_text(previous)
    with pytest.raises(ValueError, match="different evaluation implementation"):
        run_evaluation(
            Settings(_env_file=None), input_path=str(source), output_path=str(destination)
        )
    assert destination.read_text() == previous


def test_batch_retries_failed_old_adapter_rows(tmp_path):
    source = tmp_path / "input.jsonl"
    destination = tmp_path / "output.jsonl"
    source.write_text(json.dumps(SAMPLE) + "\n")
    previous = (
        json.dumps(
            {
                "question_id": "q1",
                "evaluation_status": "error",
                "evaluation_implementation": "ragas-0.4.3",
            }
        )
        + "\n"
    )
    destination.write_text(previous)
    assert (
        run_evaluation(
            Settings(_env_file=None), input_path=str(source), output_path=str(destination), limit=0
        )
        == 0
    )
    assert destination.read_text() == previous
