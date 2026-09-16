from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from elite_rag.models import RawDocument


class DocumentFormatError(ValueError):
    pass


def load_benchmark_documents(
    root: Path, *, source: str | None = None, limit: int | None = None
) -> Iterator[RawDocument]:
    """Stream EnterpriseRAG-Bench JSON documents without loading the corpus in memory."""
    sources_root = (
        root / "generated_data" / "sources" if (root / "generated_data").exists() else root
    )
    scan_root = sources_root / source.lower() if source else sources_root
    for count, path in enumerate(scan_root.rglob("*.json"), start=1):
        if limit is not None and count > limit:
            return
        relative = path.relative_to(sources_root)
        detected_source = relative.parts[0].lower()
        yield load_benchmark_document(path, detected_source)


def load_benchmark_document(path: Path, source: str | None = None) -> RawDocument:
    with path.open(encoding="utf-8") as handle:
        data: dict[str, Any] = json.load(handle)
    return load_benchmark_data(data, source or path.parent.name, str(path))


def load_benchmark_data(data: dict[str, Any], source: str, label: str = "document") -> RawDocument:
    source_name = source.lower()
    title_field = data.get("title_field_name")
    content_fields = data.get("content_field_names")
    if not isinstance(title_field, str) or title_field not in data:
        raise DocumentFormatError(f"{label}: invalid title_field_name")
    if not isinstance(content_fields, list) or not content_fields:
        raise DocumentFormatError(f"{label}: invalid content_field_names")
    missing = [name for name in content_fields if not isinstance(name, str) or name not in data]
    if missing:
        raise DocumentFormatError(f"{label}: missing declared content fields: {missing}")
    document_id = str(data.get("dataset_doc_uuid") or Path(label).stem)
    excluded = {*content_fields, "title_field_name", "content_field_names"}
    metadata = {key: value for key, value in data.items() if key not in excluded}
    return RawDocument(
        document_id=document_id,
        source=source_name,
        title=str(data[title_field]),
        fields={name: data[name] for name in content_fields},
        metadata=metadata,
    )


def load_single_document(path: Path, source: str) -> RawDocument:
    if path.suffix.lower() == ".json":
        return load_benchmark_document(path, source)
    content = path.read_text(encoding="utf-8")
    return RawDocument(
        document_id=path.stem,
        source=source,
        title=path.stem.replace("_", " ").replace("-", " ").title(),
        fields={"content": content},
        metadata={"path": str(path)},
    )
