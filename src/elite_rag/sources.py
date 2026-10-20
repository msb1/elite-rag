from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import PurePosixPath
from typing import Any

import boto3
from botocore.config import Config

from elite_rag.config import Settings
from elite_rag.documents import SourceDocument
from elite_rag.parsing import source_from_bytes


@dataclass(frozen=True, slots=True)
class ObjectRef:
    bucket: str
    key: str
    etag: str
    size_bytes: int
    source: str


class RustFSReader:
    def __init__(self, settings: Settings) -> None:
        if not settings.rustfs_access_key or not settings.rustfs_secret_key:
            raise ValueError("RustFS credentials are missing")
        self.settings = settings
        self.client: Any = boto3.client(
            "s3",
            endpoint_url=settings.rustfs_endpoint,
            aws_access_key_id=settings.rustfs_access_key,
            aws_secret_access_key=settings.rustfs_secret_key,
            region_name=settings.rustfs_region,
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        )

    def list_objects(
        self, bucket: str, prefix: str, source: str | None = None, limit: int | None = None
    ) -> Iterator[ObjectRef]:
        normalized = prefix.strip("/")
        if source and not normalized.endswith(f"/{source.strip('/')}"):
            normalized = f"{normalized}/{source.strip('/')}"
        count = 0
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=f"{normalized}/"):
            for item in page.get("Contents", []):
                key = str(item["Key"])
                if key.endswith("/") or not key.lower().endswith((".json", ".txt", ".md")):
                    continue
                if limit is not None and count >= limit:
                    return
                yield ObjectRef(
                    bucket=bucket,
                    key=key,
                    etag=str(item.get("ETag", "")).strip('"'),
                    size_bytes=int(item.get("Size", 0)),
                    source=self.infer_source(key, expected=source),
                )
                count += 1

    def head(self, bucket: str, key: str, source: str | None = None) -> ObjectRef:
        item = self.client.head_object(Bucket=bucket, Key=key)
        return ObjectRef(
            bucket,
            key,
            str(item.get("ETag", "")).strip('"'),
            int(item.get("ContentLength", 0)),
            self.infer_source(key, expected=source),
        )

    def read(self, ref: ObjectRef) -> SourceDocument:
        response = self.client.get_object(Bucket=ref.bucket, Key=ref.key)
        body = response["Body"].read()
        document = source_from_bytes(body, ref.source, ref.key)
        return replace(document, bucket=ref.bucket, etag=ref.etag)

    def infer_source(self, key: str, expected: str | None = None) -> str:
        """Resolve the canonical source from a configured path component."""
        configured = self.settings.configured_sources
        matches = [
            part.lower()
            for part in PurePosixPath(key).parts
            if part.lower() in configured
        ]
        unique_matches = tuple(dict.fromkeys(matches))
        if len(unique_matches) != 1:
            raise ValueError(
                f"Object key must contain exactly one configured source segment; "
                f"key={key!r}, matches={list(unique_matches)!r}"
            )
        detected = unique_matches[0]
        if expected and expected.strip("/").lower() != detected:
            raise ValueError(
                f"Requested source {expected!r} does not match object key source {detected!r}"
            )
        return detected
