from __future__ import annotations

import re
from collections.abc import Sequence
from threading import Lock
from typing import Any


class EmbeddingGemma:
    """OpenAI-compatible adapter matching rag-bench's CustomOpenAIEmbeddings."""

    def __init__(self, model_name: str, base_url: str, api_key: str = "lm-studio") -> None:
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self._client: Any = None
        self._lock = Lock()

    @property
    def client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    def count_tokens(self, text: str) -> int:
        # The local OpenAI-compatible endpoint does not expose its tokenizer.
        return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        with self._lock:
            response = self.client.embeddings.create(input=list(texts), model=self.model_name)
        return [list(item.embedding) for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
