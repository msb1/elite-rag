#!/usr/bin/env python3
"""Ingest hierarchical RustFS chunks with BGE-M3 and Qdrant BM25 vectors."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from elite_rag.config import Settings, get_settings
from elite_rag.ingestion import IngestionService
from elite_rag.multi import multi_settings
from elite_rag.sources import RustFSReader


def configured_sources(settings: Settings) -> list[str]:
    return [
        source.strip().lower()
        for source in settings.multi_source_types.split(",")
        if source.strip()
    ]


SOURCES_TO_INGEST = configured_sources(get_settings())


def main() -> int:
    settings = get_settings()
    try:
        settings = multi_settings(settings)
    except ValueError as exc:
        raise SystemExit(f"Invalid tri-vector configuration: {exc}") from exc

    sources = configured_sources(settings)
    if not sources:
        raise SystemExit("MULTI_SOURCE_TYPES must list at least one RustFS source")
    unknown = set(sources) - set(settings.configured_sources)
    if unknown:
        raise SystemExit(f"Sources missing from SOURCE_TYPES: {', '.join(sorted(unknown))}")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default=settings.rustfs_bucket)
    parser.add_argument("--prefix", default=settings.rustfs_prefix)
    parser.add_argument("--source", choices=sources)
    parser.add_argument(
        "--limit", type=int, help="Maximum number of objects across selected sources"
    )
    parser.add_argument("--collection", default=settings.qdrant_collection)
    parser.add_argument("--chunk-max-tokens", type=int, default=settings.multi_chunk_max_tokens)
    parser.add_argument("--report", type=Path, default=Path(settings.multi_ingestion_report))
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.chunk_max_tokens < 96:
        parser.error("--chunk-max-tokens must be at least 96")

    settings = settings.model_copy(update={
        "qdrant_collection": args.collection,
        "multi_chunk_max_tokens": args.chunk_max_tokens,
    })
    run_report: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "collection": settings.qdrant_collection,
        "configuration": {
            "pipeline_signature": settings.pipeline_signature,
            "embedding_model": settings.embedding_model,
            "embedding_dimension": settings.embedding_dimension,
            "embedding_base_url": settings.embedding_base_url,
            "tokenizer_repository": settings.tokenizer_repository,
            "chunk_target_tokens": settings.chunk_target_tokens,
            "chunk_max_tokens": settings.multi_chunk_max_tokens,
            "embedding_max_length": settings.bge_embedding_max_length,
            "chunk_overlap_tokens": settings.chunk_overlap_tokens,
            "dense_vector": settings.qdrant_dense_vector_name,
            "learned_sparse_vector": settings.qdrant_sparse_vector_name,
            "bm25_vector": settings.qdrant_bm25_vector_name,
            "bm25_model": settings.bm25_model,
            "bm25_modifier": "idf",
        },
        "documents": [],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)

    def save_report() -> None:
        temporary = args.report.with_suffix(".tmp")
        temporary.write_text(json.dumps(run_report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.report)

    save_report()
    reader = service = None
    seen = ingested = unchanged = failed = 0
    try:
        reader = RustFSReader(settings)
        service = IngestionService(settings)
        collection = service.initialize_collection()
        print(f"Collection: {collection.collection_name}", flush=True)
        selected_sources = [args.source] if args.source else sources
        for source in selected_sources:
            remaining = None if args.limit is None else args.limit - seen
            if remaining == 0:
                break
            refs = reader.list_objects(args.bucket, args.prefix, source, remaining)
            for ref in refs:
                seen += 1
                try:
                    document = reader.read(ref)
                    report = service.ingest(document)
                except Exception as exc:
                    failed += 1
                    run_report["documents"].append(
                        {
                            "object_key": ref.key,
                            "source": source,
                            "status": "failed",
                            "error": str(exc),
                        }
                    )
                    save_report()
                    print(f"FAILED {ref.key}: {exc}", file=sys.stderr, flush=True)
                    continue
                ingested += report.documents_ingested
                unchanged += report.documents_skipped_unchanged
                run_report["documents"].append(
                    {
                        "object_key": ref.key,
                        "source": source,
                        "document_id": document.document_id,
                        "source_etag": document.etag,
                        "status": "unchanged"
                        if report.documents_skipped_unchanged
                        else "empty"
                        if report.documents_skipped_empty
                        else "ready",
                        "written_passages": report.children_upserted,
                        "expected_passages": report.expected_passages,
                        "verified_passages": report.verified_passages,
                    }
                )
                save_report()
                print(f"{seen}: {ref.key} ({report.children_upserted} passages)", flush=True)
    except Exception as exc:
        run_report["status"] = "failed"
        run_report["error"] = str(exc)
        print(f"Ingestion stopped: {exc}", file=sys.stderr)
        return 1
    finally:
        if run_report["status"] == "running":
            run_report["status"] = "interrupted"
        save_report()
        if service is not None:
            service.store.client.close()
        if reader is not None:
            reader.client.close()

    run_report.update(
        status="complete" if not failed and seen else "failed",
        finished_at=datetime.now(timezone.utc).isoformat(),
        objects_seen=seen,
        documents_ingested=ingested,
        failures=failed,
        documents_unchanged=unchanged,
    )
    save_report()
    if not seen:
        print(
            f"No documents found in {args.bucket!r} under {args.prefix!r} for sources "
            f"{selected_sources!r}. Check S3_PATH_MARKDOWN_PREFIX or pass --prefix.",
            file=sys.stderr,
        )
    print(
        f"Objects seen: {seen}; documents ingested: {ingested}; "
        f"unchanged: {unchanged}; failures: {failed}"
    )
    return 1 if failed or not seen else 0


if __name__ == "__main__":
    raise SystemExit(main())
