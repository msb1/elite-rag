from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from threading import RLock
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from elite_rag.config import Settings
from elite_rag.ingestion import IngestionReport, IngestionService
from elite_rag.sources import ObjectRef, RustFSReader

LOGGER = logging.getLogger(__name__)


class IngestionJobs:
    def __init__(self, settings: Settings, service: IngestionService) -> None:
        self.settings = settings
        self.service = service
        self.tasks: set[asyncio.Task[None]] = set()
        self._memory_mode = not bool(settings.ingestion_database_url)
        self._memory_lock = RLock()
        self._memory_jobs: dict[str, dict[str, Any]] = {}
        self._memory_objects: dict[tuple[str, str, str, str], tuple[str, str]] = {}
        self._create_schema()

    def _connect(self) -> psycopg.Connection[Any]:
        return psycopg.connect(self.settings.ingestion_database_url, row_factory=dict_row)

    def _create_schema(self) -> None:
        if self._memory_mode:
            return
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS elite_ingestion_jobs (
                    job_id TEXT PRIMARY KEY,
                    bucket TEXT NOT NULL,
                    prefix TEXT NOT NULL,
                    source TEXT,
                    object_limit INTEGER,
                    replace_existing BOOLEAN NOT NULL,
                    status TEXT NOT NULL,
                    submitted_at TIMESTAMPTZ NOT NULL,
                    started_at TIMESTAMPTZ,
                    completed_at TIMESTAMPTZ,
                    report JSONB,
                    error TEXT,
                    work_items_total INTEGER NOT NULL DEFAULT 0,
                    work_items_completed INTEGER NOT NULL DEFAULT 0,
                    work_items_retryable INTEGER NOT NULL DEFAULT 0,
                    work_items_permanent_failed INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS elite_ingested_objects (
                    bucket TEXT NOT NULL,
                    object_key TEXT NOT NULL,
                    collection_name TEXT NOT NULL,
                    pipeline_version TEXT NOT NULL,
                    etag TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    completed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (bucket, object_key, collection_name, pipeline_version)
                )
                """
            )

    def submit(
        self,
        bucket: str,
        prefix: str,
        source: str | None,
        limit: int | None,
        replace_existing: bool,
    ) -> dict[str, Any]:
        if self._memory_mode:
            job_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc)
            with self._memory_lock:
                self._memory_jobs[job_id] = {
                    "job_id": job_id,
                    "status": "queued",
                    "bucket": bucket,
                    "prefix": prefix,
                    "source": source,
                    "object_limit": limit,
                    "replace_existing": replace_existing,
                    "submitted_at": now,
                    "started_at": None,
                    "completed_at": None,
                    "report": asdict(IngestionReport()),
                    "error": None,
                    "work_items_total": 0,
                    "work_items_completed": 0,
                    "work_items_retryable": 0,
                    "work_items_permanent_failed": 0,
                }
            self._start(job_id)
            result = self.get(job_id)
            assert result is not None
            return result
        # A deployment may start with the database unavailable or have its tables
        # removed while the API process remains alive. Ensure the queue schema is
        # present before attempting to persist a job.
        self._create_schema()
        job_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO elite_ingestion_jobs
                    (job_id, bucket, prefix, source, object_limit, replace_existing,
                     status, submitted_at, report)
                VALUES (%s, %s, %s, %s, %s, %s, 'queued', NOW(), %s)
                """,
                (
                    job_id,
                    bucket,
                    prefix,
                    source,
                    limit,
                    replace_existing,
                    Jsonb(asdict(IngestionReport())),
                ),
            )
        self._start(job_id)
        result = self.get(job_id)
        assert result is not None
        return result

    def resume_unfinished(self) -> None:
        if self._memory_mode:
            return
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT job_id FROM elite_ingestion_jobs WHERE status IN ('queued', 'running')"
            ).fetchall()
        for row in rows:
            self._start(row["job_id"])

    def _start(self, job_id: str) -> None:
        task = asyncio.create_task(asyncio.to_thread(self._run, job_id))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    def get(self, job_id: str) -> dict[str, Any] | None:
        if self._memory_mode:
            with self._memory_lock:
                row = self._memory_jobs.get(job_id)
                return dict(row) if row is not None else None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM elite_ingestion_jobs WHERE job_id = %s", (job_id,)
            ).fetchone()
        if row is None:
            return None
        return {
            "job_id": row["job_id"],
            "status": row["status"],
            "bucket": row["bucket"],
            "prefix": row["prefix"],
            "submitted_at": row["submitted_at"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "report": row["report"],
            "error": row["error"],
            "work_items_total": row["work_items_total"],
            "work_items_completed": row["work_items_completed"],
            "work_items_retryable": row["work_items_retryable"],
            "work_items_permanent_failed": row["work_items_permanent_failed"],
        }

    def is_current(self, ref: ObjectRef) -> bool:
        if self._memory_mode:
            key = (
                ref.bucket,
                ref.key,
                self.settings.qdrant_collection,
                self.settings.pipeline_signature,
            )
            with self._memory_lock:
                current = self._memory_objects.get(key)
            return bool(current and current[0] == ref.etag)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT etag FROM elite_ingested_objects
                WHERE bucket = %s AND object_key = %s AND collection_name = %s
                  AND pipeline_version = %s
                """,
                (
                    ref.bucket,
                    ref.key,
                    self.settings.qdrant_collection,
                    self.settings.pipeline_signature,
                ),
            ).fetchone()
        return bool(row and row["etag"] == ref.etag)

    def record_current(self, ref: ObjectRef, document_id: str) -> None:
        if self._memory_mode:
            key = (
                ref.bucket,
                ref.key,
                self.settings.qdrant_collection,
                self.settings.pipeline_signature,
            )
            with self._memory_lock:
                self._memory_objects[key] = (ref.etag, document_id)
            return
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO elite_ingested_objects
                    (bucket, object_key, collection_name, pipeline_version, etag, document_id)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (bucket, object_key, collection_name, pipeline_version)
                DO UPDATE SET etag = EXCLUDED.etag, document_id = EXCLUDED.document_id,
                              completed_at = NOW()
                """,
                (
                    ref.bucket,
                    ref.key,
                    self.settings.qdrant_collection,
                    self.settings.pipeline_signature,
                    ref.etag,
                    document_id,
                ),
            )

    def clear_current_collection(self) -> None:
        if self._memory_mode:
            with self._memory_lock:
                self._memory_objects = {
                    key: value
                    for key, value in self._memory_objects.items()
                    if key[2] != self.settings.qdrant_collection
                }
            return
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM elite_ingested_objects WHERE collection_name = %s",
                (self.settings.qdrant_collection,),
            )

    def _run(self, job_id: str) -> None:
        if self._memory_mode:
            try:
                self._process_memory(job_id)
            except Exception as exc:
                LOGGER.exception("prefix_job_failed job_id=%s", job_id)
                self._update(
                    job_id, status="failed", error=str(exc), completed_at=datetime.now(timezone.utc)
                )
            return
        # The advisory lock prevents two API workers from processing the same resumed job.
        with self._connect() as lock_conn:
            lock = lock_conn.execute(
                "SELECT pg_try_advisory_lock(hashtext(%s)) AS acquired", (job_id,)
            ).fetchone()
            if lock is None or not lock["acquired"]:
                return
            try:
                self._process(job_id)
            except Exception as exc:
                LOGGER.exception("prefix_job_failed job_id=%s", job_id)
                self._update(
                    job_id, status="failed", error=str(exc), completed_at=datetime.now(timezone.utc)
                )
            finally:
                lock_conn.execute("SELECT pg_advisory_unlock(hashtext(%s))", (job_id,))

    def _process(self, job_id: str) -> None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM elite_ingestion_jobs WHERE job_id = %s", (job_id,)
            ).fetchone()
        if row is None or row["status"] == "completed":
            return
        assert row is not None
        self._update(job_id, status="running", started_at=datetime.now(timezone.utc), error=None)
        reader = RustFSReader(self.settings)
        report = IngestionReport()
        seen = 0
        completed = 0
        failed = 0
        for ref in reader.list_objects(
            row["bucket"], row["prefix"], row["source"], row["object_limit"]
        ):
            seen += 1
            if self.is_current(ref) and not row["replace_existing"]:
                completed += 1
                self._progress(job_id, report, seen, completed, failed)
                continue
            try:
                document = reader.read(ref)
            except (ValueError, UnicodeError, json.JSONDecodeError):
                LOGGER.exception("prefix_object_invalid job_id=%s key=%s", job_id, ref.key)
                failed += 1
                report.documents_seen += 1
                report.documents_skipped_failed += 1
                self._progress(job_id, report, seen, completed, failed)
                continue
            one = self.service.ingest(document, row["replace_existing"])
            report.add(one)
            self.record_current(ref, document.document_id)
            completed += 1
            self._progress(job_id, report, seen, completed, failed)
        self._update(job_id, status="completed", completed_at=datetime.now(timezone.utc))

    def _process_memory(self, job_id: str) -> None:
        row = self.get(job_id)
        if row is None:
            return
        self._update(job_id, status="running", started_at=datetime.now(timezone.utc), error=None)
        reader = RustFSReader(self.settings)
        report = IngestionReport()
        seen = completed = failed = 0
        for ref in reader.list_objects(
            row["bucket"], row["prefix"], row["source"], row["object_limit"]
        ):
            seen += 1
            if self.is_current(ref) and not row["replace_existing"]:
                completed += 1
                self._progress(job_id, report, seen, completed, failed)
                continue
            try:
                document = reader.read(ref)
            except (ValueError, UnicodeError, json.JSONDecodeError):
                LOGGER.exception("prefix_object_invalid job_id=%s key=%s", job_id, ref.key)
                failed += 1
                report.documents_seen += 1
                report.documents_skipped_failed += 1
                self._progress(job_id, report, seen, completed, failed)
                continue
            one = self.service.ingest(document, row["replace_existing"])
            report.add(one)
            self.record_current(ref, document.document_id)
            completed += 1
            self._progress(job_id, report, seen, completed, failed)
        self._update(job_id, status="completed", completed_at=datetime.now(timezone.utc))

    def _progress(
        self, job_id: str, report: IngestionReport, seen: int, completed: int, failed: int
    ) -> None:
        self._update(
            job_id,
            report=asdict(report),
            work_items_total=seen,
            work_items_completed=completed,
            work_items_permanent_failed=failed,
        )

    def _update(self, job_id: str, **fields: Any) -> None:
        if not fields:
            return
        if self._memory_mode:
            with self._memory_lock:
                row = self._memory_jobs.get(job_id)
                if row is not None:
                    row.update(fields)
            return
        assignments = ", ".join(f"{key} = %s" for key in fields)
        values = tuple(
            Jsonb(value) if isinstance(value, dict) else value for value in fields.values()
        )
        with self._connect() as conn:
            conn.execute(
                f"UPDATE elite_ingestion_jobs SET {assignments} WHERE job_id = %s",
                (*values, job_id),
            )
