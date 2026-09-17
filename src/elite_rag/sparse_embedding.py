from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import requests

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SparseVector:
    indices: list[int]
    values: list[float]


class MiniCOILSparseEmbedder:
    """HTTP client for the remote bounded-memory FastEmbed miniCOIL service."""

    def __init__(
        self,
        endpoint_url: str,
        model_name: str,
        batch_size: int = 8,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.endpoint_url = endpoint_url
        self.model_name = model_name
        self.batch_size = batch_size
        self.timeout_seconds = timeout_seconds

    def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        return self._embed(texts, "document")

    def embed_query(self, text: str) -> SparseVector:
        return self._embed([text], "query")[0]

    def _embed(self, texts: Sequence[str], mode: str) -> list[SparseVector]:
        if not texts:
            return []
        vectors: list[SparseVector] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            LOGGER.info("minicoil_remote_request_started mode=%s texts=%s", mode, len(batch))
            response = requests.post(
                self.endpoint_url,
                headers={"Content-Type": "application/json"},
                json={"model": self.model_name, "mode": mode, "texts": batch},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            raw_vectors = payload.get("vectors", [])
            if len(raw_vectors) != len(batch):
                raise ValueError(
                    "miniCOIL service returned a different number of vectors than inputs"
                )
            vectors.extend(
                SparseVector(
                    indices=[int(index) for index in item["indices"]],
                    values=[float(value) for value in item["values"]],
                )
                for item in raw_vectors
            )
            LOGGER.info("minicoil_remote_request_completed mode=%s texts=%s", mode, len(batch))
        return vectors
