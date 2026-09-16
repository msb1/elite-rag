from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from elite_rag.intent import DeterministicIntentRouter, fallback_query
from elite_rag.retrieval import deduplicate_parents


def router(tmp_path: Path) -> DeterministicIntentRouter:
    aliases = tmp_path / "aliases.json"
    aliases.write_text(
        json.dumps(
            {
                "sources": {"jira": "jira", "ticket": ["jira", "linear"], "slack": "slack"},
                "projects": {"hydra": "hydra"},
            }
        ),
        encoding="utf-8",
    )
    return DeterministicIntentRouter(aliases)


def test_extracts_multiple_sources_and_explicit_project(tmp_path: Path) -> None:
    result = router(tmp_path).extract("Compare Jira with Slack for project:Hydra")
    assert result is not None
    assert result.sources == ("jira", "slack")
    assert result.projects == ("hydra",)


def test_alias_matching_does_not_match_substrings(tmp_path: Path) -> None:
    assert router(tmp_path).extract("The slackened rope") is None


def test_ambiguous_alias_expands_to_all_valid_sources(tmp_path: Path) -> None:
    result = router(tmp_path).extract("Find the ticket")
    assert result is not None
    assert result.sources == ("jira", "linear")


def test_fallback_removes_only_noise() -> None:
    assert (
        fallback_query("What exactly caused the token leak in production?")
        == "exactly caused token leak production"
    )

    assert fallback_query("Find source:jira project:hydra auth failures") == "auth failures"


def test_parent_dedup_keeps_first_vector_score() -> None:
    payload = {
        "parent_id": "p1",
        "document_id": "d1",
        "parent_context": "parent",
        "metadata_filters": {"source": "jira"},
    }
    hits = [
        SimpleNamespace(payload=payload, score=0.9),
        SimpleNamespace(payload=payload, score=0.8),
    ]
    parents = deduplicate_parents(hits)
    assert len(parents) == 1
    assert parents[0].vector_score == 0.9
