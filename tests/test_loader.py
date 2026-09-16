from __future__ import annotations

import json
from pathlib import Path

import pytest

from elite_rag.loaders import DocumentFormatError, load_benchmark_document, load_benchmark_documents


def test_loads_declared_benchmark_fields(tmp_path: Path) -> None:
    path = tmp_path / "generated_data" / "sources" / "jira" / "team" / "ticket.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "summary": "Broken build",
                "description": "Details",
                "comments": ["first", "second"],
                "project": "CORE",
                "updated_at": "2026-09-10",
                "dataset_doc_uuid": "dsid_123",
                "title_field_name": "summary",
                "content_field_names": ["description", "comments"],
            }
        ),
        encoding="utf-8",
    )
    loaded = list(load_benchmark_documents(tmp_path))
    assert len(loaded) == 1
    assert loaded[0].source == "jira"
    assert loaded[0].title == "Broken build"
    assert loaded[0].fields == {"description": "Details", "comments": ["first", "second"]}
    assert loaded[0].metadata["project"] == "CORE"


def test_rejects_missing_declared_field(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {"title": "Bad", "title_field_name": "title", "content_field_names": ["missing"]}
        ),
        encoding="utf-8",
    )
    with pytest.raises(DocumentFormatError, match="missing declared"):
        load_benchmark_document(path, "jira")
