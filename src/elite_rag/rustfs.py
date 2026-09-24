from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from elite_rag.config import Settings
from elite_rag.models import RawDocument

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RustFSObject:
    key: str
    etag: str | None
    size_bytes: int | None


class RustFSListingError(RuntimeError):
    """Raised when RustFS returns an internally inconsistent object listing."""


def create_rustfs_client(settings: Settings) -> Any:
    import boto3
    from botocore.config import Config

    if not settings.rustfs_access_key or not settings.rustfs_secret_key:
        raise ValueError("RustFS credentials are missing; configure the project .env")
    return boto3.client(
        "s3",
        endpoint_url=settings.rustfs_endpoint,
        aws_access_key_id=settings.rustfs_access_key,
        aws_secret_access_key=settings.rustfs_secret_key,
        region_name=settings.rustfs_region,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def list_rustfs_child_prefixes(client: Any, bucket: str, prefix: str) -> tuple[str, ...]:
    """Return immediate child prefixes below ``prefix`` without recursive listing.

    A full-corpus inventory deliberately enumerates every immediate source prefix
    independently. This avoids relying on one continuation-token sequence across
    a large mixed-source prefix.
    """
    root = prefix.strip("/")
    paginator = client.get_paginator("list_objects_v2")
    prefixes: set[str] = set()
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{root}/", Delimiter="/"):
        for entry in page.get("CommonPrefixes", []):
            child = str(entry["Prefix"]).removeprefix(f"{root}/").rstrip("/")
            if child:
                prefixes.add(child)
    discovered = tuple(sorted(prefixes))
    LOGGER.info(
        "rustfs_child_prefixes_discovered bucket=%s prefix=%s count=%s prefixes=%s",
        bucket,
        root,
        len(discovered),
        list(discovered),
    )
    return discovered


def iter_rustfs_documents(
    client: Any,
    bucket: str,
    prefix: str = "documents",
    source: str | None = None,
    limit: int | None = None,
) -> Iterator[RawDocument]:
    """Stream JSON or exported TXT documents from RustFS without local staging."""
    root = prefix.strip("/")
    if source:
        root = f"{root}/{source.strip('/')}"
    LOGGER.info("rustfs_prefix_listing bucket=%s prefix=%s", bucket, root)
    paginator = client.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{root}/"):
        for item in page.get("Contents", []):
            key = str(item["Key"])
            if key.endswith("/"):
                continue
            if limit is not None and count >= limit:
                LOGGER.info(
                    "rustfs_prefix_limit_reached bucket=%s prefix=%s limit=%s", bucket, root, limit
                )
                return
            LOGGER.info(
                "rustfs_object_read_started bucket=%s key=%s size=%s",
                bucket,
                key,
                item.get("Size"),
            )
            try:
                response = client.get_object(Bucket=bucket, Key=key)
                body = response["Body"].read()
                document = document_from_rustfs_object(key, body, root, source)
            except Exception:
                LOGGER.exception("rustfs_object_read_failed bucket=%s key=%s", bucket, key)
                raise
            LOGGER.info(
                "rustfs_object_read_completed bucket=%s key=%s bytes=%s document_id=%s source=%s",
                bucket,
                key,
                len(body),
                document.document_id,
                document.source,
            )
            yield document
            count += 1


def iter_rustfs_objects(
    client: Any,
    bucket: str,
    prefix: str = "documents",
    source: str | None = None,
    limit: int | None = None,
) -> Iterator[RustFSObject]:
    """List ingestible RustFS objects without downloading their contents."""
    root = prefix.strip("/")
    if source:
        root = f"{root}/{source.strip('/')}"
    LOGGER.info("rustfs_prefix_listing bucket=%s prefix=%s", bucket, root)
    paginator = client.get_paginator("list_objects_v2")
    count = 0
    page_count = 0
    last_key: str | None = None
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{root}/"):
        page_count += 1
        for item in page.get("Contents", []):
            key = str(item["Key"])
            if key.endswith("/"):
                continue
            if limit is not None and count >= limit:
                LOGGER.info(
                    "rustfs_prefix_limit_reached bucket=%s prefix=%s limit=%s", bucket, root, limit
                )
                return
            yield RustFSObject(
                key=key,
                etag=str(item["ETag"]).strip('"') if item.get("ETag") else None,
                size_bytes=int(item["Size"]) if item.get("Size") is not None else None,
            )
            count += 1
            last_key = key
    _validate_rustfs_listing_end(client, bucket, root, last_key)
    LOGGER.info(
        "rustfs_source_listing_completed bucket=%s prefix=%s objects=%s pages=%s",
        bucket,
        root,
        count,
        page_count,
    )


def _validate_rustfs_listing_end(
    client: Any, bucket: str, root: str, last_key: str | None
) -> None:
    """Detect a paginator that stopped while a source prefix still has objects."""
    if last_key is None:
        return
    response = client.list_objects_v2(
        Bucket=bucket,
        Prefix=f"{root}/",
        StartAfter=last_key,
        MaxKeys=1,
    )
    remaining = next(
        (
            str(item["Key"])
            for item in response.get("Contents", [])
            if not str(item["Key"]).endswith("/")
        ),
        None,
    )
    if remaining is None:
        return
    message = (
        "RustFS source listing ended before all objects were enumerated "
        f"(prefix={root!r}, last_key={last_key!r}, next_key={remaining!r})"
    )
    LOGGER.error("rustfs_source_listing_inconsistent %s", message)
    raise RustFSListingError(message)


def read_rustfs_document(
    client: Any, bucket: str, key: str, source_hint: str | None = None
) -> RawDocument:
    """Load one JSON or text object from RustFS into a pipeline document."""
    LOGGER.info("rustfs_object_read_started bucket=%s key=%s", bucket, key)
    try:
        response = client.get_object(Bucket=bucket, Key=key)
        body = response["Body"].read()
        document = document_from_rustfs_object(key, body, "", source_hint)
    except Exception:
        LOGGER.exception("rustfs_object_read_failed bucket=%s key=%s", bucket, key)
        raise
    LOGGER.info(
        "rustfs_object_read_completed bucket=%s key=%s bytes=%s document_id=%s source=%s",
        bucket,
        key,
        len(body),
        document.document_id,
        document.source,
    )
    return document


def head_rustfs_object(client: Any, bucket: str, key: str) -> RustFSObject:
    """Read RustFS object identity without downloading its body."""
    response = client.head_object(Bucket=bucket, Key=key)
    return RustFSObject(
        key=key,
        etag=str(response["ETag"]).strip('"') if response.get("ETag") else None,
        size_bytes=int(response["ContentLength"]) if response.get("ContentLength") else 0,
    )


def document_from_rustfs_object(
    key: str, body: bytes, prefix: str, source_hint: str | None = None
) -> RawDocument:
    import json
    import re

    path = PurePosixPath(key)
    source = source_hint or (
        path.parts[len(PurePosixPath(prefix).parts)]
        if len(path.parts) > len(PurePosixPath(prefix).parts)
        else "unknown"
    )
    if path.suffix.lower() == ".json":
        from elite_rag.loaders import load_benchmark_data

        data = json.loads(body)
        return load_benchmark_data(data, source, key)
    text = body.decode("utf-8")
    lines = text.splitlines()
    title = lines[0].strip() if lines else path.stem
    document_id_match = re.search(r"dsid_[0-9a-f]{32}", path.name)
    document_id = document_id_match.group(0) if document_id_match else path.stem
    return RawDocument(
        document_id=document_id,
        source=source,
        title=title or path.stem,
        fields={"content": "\n".join(lines[1:]).strip() if lines else ""},
        metadata={"object_key": key},
    )
