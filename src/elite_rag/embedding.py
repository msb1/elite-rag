from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from openai import OpenAI


@dataclass(frozen=True, slots=True)
class HybridVector:
    dense: list[float]
    sparse_indices: list[int]
    sparse_values: list[float]


class HybridEmbedder:
    """Client for remote-models.py's BGE-M3 OpenAI-shaped embedding endpoint."""

    def __init__(self, model: str, dimension: int, base_url: str, api_key: str,
                 timeout: float = 120.0) -> None:
        self.model = model
        self.dimension = dimension
        self.url = f"{base_url.rstrip('/')}/embeddings"
        self.api_key = api_key
        self.timeout = timeout

    def embed_documents(self, texts: Sequence[str]) -> list[HybridVector]:
        return self._embed(texts)

    def embed_queries(self, texts: Sequence[str]) -> list[HybridVector]:
        return self._embed(texts)

    def _embed(self, texts: Sequence[str]) -> list[HybridVector]:
        if not texts:
            return []
        request = Request(
            self.url,
            data=json.dumps({"model": self.model, "input": list(texts)}).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"embedding endpoint HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(f"embedding endpoint request failed: {exc}") from exc
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("embedding endpoint returned invalid JSON") from exc
        items = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(items, list) or len(items) != len(texts):
            raise ValueError("embedding endpoint returned the wrong number of vectors")
        ordered: list[HybridVector | None] = [None] * len(texts)
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("embedding endpoint returned an invalid item")
            index = item.get("index")
            if type(index) is not int or not 0 <= index < len(texts) or ordered[index] is not None:
                raise ValueError("embedding endpoint returned missing or duplicate input indices")
            dense = item.get("embedding")
            sparse = item.get("sparse_vector")
            if not isinstance(dense, list) or len(dense) != self.dimension or not all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) for value in dense
            ):
                raise ValueError("embedding endpoint returned an invalid dense vector")
            if not isinstance(sparse, dict):
                raise ValueError("embedding endpoint did not return a sparse vector")
            indices, values = sparse.get("indices"), sparse.get("values")
            if (not isinstance(indices, list) or not isinstance(values, list)
                or len(indices) != len(values)
                or any(type(value) is not int or value < 0 for value in indices)
                or len(set(indices)) != len(indices)
                or any(not isinstance(value, (int, float)) or isinstance(value, bool)
                       or not math.isfinite(value) for value in values)):
                raise ValueError("embedding endpoint returned an invalid sparse vector")
            ordered[index] = HybridVector(
                [float(value) for value in dense], indices,
                [float(value) for value in values],
            )
        if any(vector is None for vector in ordered):
            raise ValueError("embedding endpoint returned missing input indices")
        return [vector for vector in ordered if vector is not None]


class DenseEmbedder:
    def __init__(self, model: str, dimension: int, base_url: str, api_key: str) -> None:
        self.model = model
        self.dimension = dimension
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed retrieval queries with the remote embedding model."""
        return self._embed(texts, input_type="query")

    def _embed(
        self, texts: Sequence[str], input_type: str | None = None
    ) -> list[list[float]]:
        if not texts:
            return []
        if input_type is not None:
            response = self.client.embeddings.create(
                model=self.model,
                input=list(texts),
                extra_body={"input_type": input_type},
            )
        else:
            response = self.client.embeddings.create(model=self.model, input=list(texts))
        ordered = sorted(response.data, key=lambda item: item.index)
        if len(ordered) != len(texts) or [item.index for item in ordered] != list(
            range(len(texts))
        ):
            raise ValueError("Embedding endpoint returned missing or duplicate input indices")
        vectors = [list(item.embedding) for item in ordered]
        if any(len(vector) != self.dimension for vector in vectors):
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}; "
                f"received {[len(vector) for vector in vectors]}"
            )
        return vectors
