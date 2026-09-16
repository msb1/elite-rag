from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

from elite_rag.models import RawDocument


@dataclass(frozen=True, slots=True)
class MissingDocument:
    """A source document skipped after a document-specific ingestion failure."""

    document_id: str
    source: str
    title: str
    object_key: str | None
    error_type: str
    error: str
    recorded_at: str


class MissingDocumentRecorder:
    """Maintain an atomic JSON record of documents skipped during ingestion."""

    def __init__(self, file_path: str) -> None:
        self.path = Path(file_path)
        self._lock = Lock()

    def record(self, document: RawDocument, exc: Exception) -> MissingDocument:
        entry = MissingDocument(
            document_id=document.document_id,
            source=document.source,
            title=document.title,
            object_key=_object_key(document.metadata),
            error_type=type(exc).__name__,
            error=str(exc),
            recorded_at=datetime.now(UTC).isoformat(),
        )
        with self._lock:
            entries = self._read_entries()
            entries.append(asdict(entry))
            self._write_entries(entries)
        return entry

    def _read_entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Missing-document file '{self.path}' must contain a JSON array")
        return [dict(entry) for entry in payload if isinstance(entry, dict)]

    def _write_entries(self, entries: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary_path.write_text(
            json.dumps(entries, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary_path.replace(self.path)


def _object_key(metadata: dict[str, Any]) -> str | None:
    key = metadata.get("object_key")
    return str(key) if key is not None else None
