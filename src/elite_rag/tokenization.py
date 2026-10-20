from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer


class TokenCounter:
    def __init__(self, repository: str, tokenizer_json_path: str | None = None) -> None:
        path = tokenizer_json_path or hf_hub_download(repository, "tokenizer.json")
        self.tokenizer = Tokenizer.from_file(str(Path(path)))

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)
