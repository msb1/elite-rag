from __future__ import annotations

import pytest

from elite_rag.rustfs import (
    RustFSListingError,
    iter_rustfs_objects,
    list_rustfs_child_prefixes,
)


class FakePaginator:
    def __init__(self, pages: list[dict[str, object]]) -> None:
        self.pages = pages

    def paginate(self, **_: object) -> list[dict[str, object]]:
        return self.pages


class FakeRustFSClient:
    def __init__(
        self, pages: list[dict[str, object]], remaining: list[dict[str, object]] | None = None
    ) -> None:
        self.pages = pages
        self.remaining = remaining or []

    def get_paginator(self, _: str) -> FakePaginator:
        return FakePaginator(self.pages)

    def list_objects_v2(self, **_: object) -> dict[str, object]:
        return {"Contents": self.remaining}


def test_child_prefixes_are_discovered_without_recursive_listing() -> None:
    client = FakeRustFSClient(
        [{"CommonPrefixes": [{"Prefix": "docs/gmail/"}, {"Prefix": "docs/slack/"}]}]
    )

    assert list_rustfs_child_prefixes(client, "enterprise-rag", "docs") == ("gmail", "slack")


def test_source_listing_rejects_premature_paginator_completion() -> None:
    client = FakeRustFSClient(
        [{"Contents": [{"Key": "docs/gmail/first.txt", "ETag": "etag", "Size": 12}]}],
        remaining=[{"Key": "docs/gmail/second.txt"}],
    )

    with pytest.raises(RustFSListingError, match="ended before all objects"):
        list(iter_rustfs_objects(client, "enterprise-rag", "docs", "gmail"))
