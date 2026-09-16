from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TypeAlias

from elite_rag.vector_store import SearchFilter

EXPLICIT_FILTER = re.compile(r"\b(source|project):(?:\"([^\"]+)\"|'([^']+)'|([\w.-]+))", re.I)
AliasValue: TypeAlias = str | list[str]
DEFAULT_ALIASES = {
    "sources": {
        "chat": "slack",
        "confluence": "confluence",
        "drive": "google_drive",
        "email": "gmail",
        "fireflies": "fireflies",
        "github": "github",
        "gmail": "gmail",
        "hubspot": "hubspot",
        "jira": "jira",
        "linear": "linear",
        "mail": "gmail",
        "meeting": "fireflies",
        "pull request": "github",
        "slack": "slack",
        "ticket": ["jira", "linear"],
        "wiki": "confluence",
    },
    "projects": {"devops": "devops", "hydra": "hydra", "sre": "sre"},
}


class DeterministicIntentRouter:
    def __init__(self, aliases_path: Path | None = None) -> None:
        path = aliases_path or Path("configs/filter_aliases.json")
        aliases = json.loads(path.read_text(encoding="utf-8")) if path.exists() else DEFAULT_ALIASES
        self.source_aliases = self._validate_aliases(aliases.get("sources", {}))
        self.project_aliases = self._validate_aliases(aliases.get("projects", {}))

    def extract(self, query: str) -> SearchFilter | None:
        normalized = query.casefold()
        explicit_sources: set[str] = set()
        explicit_projects: set[str] = set()
        for match in EXPLICIT_FILTER.finditer(normalized):
            value = next(group for group in match.groups()[1:] if group is not None)
            if match.group(1).lower() == "source":
                explicit_sources.update(self._resolve(value, self.source_aliases))
            else:
                explicit_projects.update(self._resolve(value, self.project_aliases))

        sources = explicit_sources or self._match_aliases(normalized, self.source_aliases)
        projects = explicit_projects or self._match_aliases(normalized, self.project_aliases)
        result = SearchFilter(tuple(sorted(sources)), tuple(sorted(projects)))
        return result if result.active else None

    @staticmethod
    def _match_aliases(text: str, aliases: dict[str, tuple[str, ...]]) -> set[str]:
        matched: set[str] = set()
        for alias, canonical_values in aliases.items():
            if re.search(rf"(?<!\w){re.escape(alias.casefold())}(?!\w)", text):
                matched.update(canonical_values)
        return matched

    @staticmethod
    def _resolve(value: str, aliases: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
        return aliases.get(value, (value,))

    @staticmethod
    def _validate_aliases(raw_aliases: object) -> dict[str, tuple[str, ...]]:
        if not isinstance(raw_aliases, dict):
            raise ValueError("Filter aliases must be a JSON object")
        validated: dict[str, tuple[str, ...]] = {}
        for raw_alias, raw_value in raw_aliases.items():
            alias = str(raw_alias).casefold()
            if isinstance(raw_value, str):
                values = (raw_value.casefold(),)
            elif isinstance(raw_value, list) and all(isinstance(item, str) for item in raw_value):
                values = tuple(item.casefold() for item in raw_value)
            else:
                raise ValueError(f"Alias '{alias}' must map to a string or list of strings")
            validated[alias] = values
        return validated


STOP_WORDS = {
    "a",
    "about",
    "are",
    "did",
    "find",
    "for",
    "how",
    "in",
    "is",
    "me",
    "of",
    "on",
    "regarding",
    "show",
    "the",
    "to",
    "was",
    "what",
    "why",
}


def fallback_query(query: str) -> str:
    query_without_filters = EXPLICIT_FILTER.sub(" ", query)
    words = re.findall(r"[\w.-]+", query_without_filters.casefold())
    filtered = [word for word in words if word not in STOP_WORDS]
    return " ".join(filtered) or query.strip()
