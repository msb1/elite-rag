"""PostgreSQL-backed durable state for RustFS prefix ingestion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

import psycopg
from psycopg.rows import dict_row

from elite_rag.ingestion import IngestionReport


@dataclass(frozen=True, slots=True)
class DurableIngestionRun:
    job_id: str
    bucket: str
    prefix: str
    source: str | None
    limit: int | None
    replace_existing: bool
    status: str
    submitted_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    error: str | None
    report: IngestionReport
    work_items_total: int
    work_items_completed: int
    work_items_retryable: int
    work_items_permanent_failed: int

    def response(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "bucket": self.bucket,
            "prefix": self.prefix,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "report": asdict(self.report),
            "error": self.error,
            "work_items_total": self.work_items_total,
            "work_items_completed": self.work_items_completed,
            "work_items_retryable": self.work_items_retryable,
            "work_items_permanent_failed": self.work_items_permanent_failed,
        }


@dataclass(frozen=True, slots=True)
class IngestionWorkItem:
    item_id: int
    job_id: str
    bucket: str
    key: str
    source: str | None
    etag: str | None
    size_bytes: int | None
    attempts: int


class IngestionLedger:
    """Transactional ledger that makes object-level ingestion restartable."""

    def __init__(self, database_url: str, lease_seconds: int = 900) -> None:
        self.database_url = database_url
        self.lease_seconds = lease_seconds

    def ensure_schema(self) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS ingestion_runs (
                    job_id UUID PRIMARY KEY,
                    bucket TEXT NOT NULL,
                    prefix TEXT NOT NULL,
                    source TEXT NULL,
                    object_limit INTEGER NULL,
                    replace_existing BOOLEAN NOT NULL,
                    collection_name TEXT NOT NULL,
                    pipeline_version TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('queued', 'running', 'completed', 'failed')
                    ),
                    submitted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    started_at TIMESTAMPTZ NULL,
                    completed_at TIMESTAMPTZ NULL,
                    error TEXT NULL
                );

                CREATE TABLE IF NOT EXISTS ingestion_work_items (
                    item_id BIGSERIAL PRIMARY KEY,
                    job_id UUID NOT NULL REFERENCES ingestion_runs(job_id) ON DELETE CASCADE,
                    bucket TEXT NOT NULL,
                    object_key TEXT NOT NULL,
                    source TEXT NULL,
                    etag TEXT NULL,
                    size_bytes BIGINT NULL,
                    status TEXT NOT NULL CHECK (status IN (
                        'pending', 'processing', 'completed', 'retryable_failure',
                        'permanent_failure'
                    )),
                    attempts INTEGER NOT NULL DEFAULT 0,
                    lease_owner UUID NULL,
                    lease_expires_at TIMESTAMPTZ NULL,
                    error_type TEXT NULL,
                    error TEXT NULL,
                    started_at TIMESTAMPTZ NULL,
                    completed_at TIMESTAMPTZ NULL,
                    documents_seen INTEGER NOT NULL DEFAULT 0,
                    documents_ingested INTEGER NOT NULL DEFAULT 0,
                    documents_skipped_empty INTEGER NOT NULL DEFAULT 0,
                    documents_skipped_failed INTEGER NOT NULL DEFAULT 0,
                    parents_created INTEGER NOT NULL DEFAULT 0,
                    children_upserted INTEGER NOT NULL DEFAULT 0,
                    batches_upserted INTEGER NOT NULL DEFAULT 0,
                    documents_replaced INTEGER NOT NULL DEFAULT 0,
                    UNIQUE (job_id, bucket, object_key)
                );

                CREATE INDEX IF NOT EXISTS ingestion_work_items_claim_idx
                    ON ingestion_work_items (job_id, status, item_id);
                CREATE INDEX IF NOT EXISTS ingestion_work_items_lease_idx
                    ON ingestion_work_items (lease_expires_at)
                    WHERE status = 'processing';
                """
            )

    def create_or_resume_run(
        self,
        *,
        bucket: str,
        prefix: str,
        source: str | None,
        limit: int | None,
        replace_existing: bool,
        collection_name: str,
        pipeline_version: str,
    ) -> DurableIngestionRun:
        with self._connect() as connection, connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT job_id FROM ingestion_runs
                WHERE bucket = %s
                  AND prefix = %s
                  AND source IS NOT DISTINCT FROM %s
                  AND object_limit IS NOT DISTINCT FROM %s
                  AND replace_existing = %s
                  AND collection_name = %s
                  AND pipeline_version = %s
                  AND status IN ('queued', 'running', 'failed')
                ORDER BY submitted_at DESC
                LIMIT 1
                FOR UPDATE
                """,
                (
                    bucket,
                    prefix,
                    source,
                    limit,
                    replace_existing,
                    collection_name,
                    pipeline_version,
                ),
            )
            existing = cursor.fetchone()
            if existing is None:
                job_id = uuid4()
                cursor.execute(
                    """
                    INSERT INTO ingestion_runs (
                        job_id, bucket, prefix, source, object_limit, replace_existing,
                        collection_name, pipeline_version, status
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'queued')
                    """,
                    (
                        job_id,
                        bucket,
                        prefix,
                        source,
                        limit,
                        replace_existing,
                        collection_name,
                        pipeline_version,
                    ),
                )
            else:
                job_id = existing["job_id"]
                cursor.execute(
                    """
                    UPDATE ingestion_runs
                    SET status = 'queued', error = NULL, completed_at = NULL
                    WHERE job_id = %s
                    """,
                    (job_id,),
                )
                cursor.execute(
                    """
                    UPDATE ingestion_work_items
                    SET status = 'pending', error_type = NULL, error = NULL,
                        lease_owner = NULL, lease_expires_at = NULL
                    WHERE job_id = %s AND status = 'retryable_failure'
                    """,
                    (job_id,),
                )
        run = self.get_run(str(job_id))
        if run is None:
            raise RuntimeError("Newly created ingestion run could not be read")
        return run

    def get_run(self, job_id: str) -> DurableIngestionRun | None:
        with self._connect() as connection, connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute("SELECT * FROM ingestion_runs WHERE job_id = %s", (UUID(job_id),))
            run = cursor.fetchone()
            if run is None:
                return None
            cursor.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    COUNT(*) FILTER (WHERE status = 'completed') AS completed,
                    COUNT(*) FILTER (WHERE status = 'retryable_failure') AS retryable,
                    COUNT(*) FILTER (WHERE status = 'permanent_failure') AS permanent_failed,
                    COALESCE(SUM(documents_seen), 0) AS documents_seen,
                    COALESCE(SUM(documents_ingested), 0) AS documents_ingested,
                    COALESCE(SUM(documents_skipped_empty), 0) AS documents_skipped_empty,
                    COALESCE(SUM(documents_skipped_failed), 0) AS documents_skipped_failed,
                    COALESCE(SUM(parents_created), 0) AS parents_created,
                    COALESCE(SUM(children_upserted), 0) AS children_upserted,
                    COALESCE(SUM(batches_upserted), 0) AS batches_upserted,
                    COALESCE(SUM(documents_replaced), 0) AS documents_replaced
                FROM ingestion_work_items
                WHERE job_id = %s
                """,
                (run["job_id"],),
            )
            summary = cursor.fetchone()
        if summary is None:
            raise RuntimeError(f"Ingestion run '{job_id}' has no work-item summary")
        report = IngestionReport(
            documents_seen=int(summary["documents_seen"]),
            documents_ingested=int(summary["documents_ingested"]),
            documents_skipped_empty=int(summary["documents_skipped_empty"]),
            documents_skipped_failed=int(summary["documents_skipped_failed"]),
            parents_created=int(summary["parents_created"]),
            children_upserted=int(summary["children_upserted"]),
            batches_upserted=int(summary["batches_upserted"]),
            documents_replaced=int(summary["documents_replaced"]),
        )
        return DurableIngestionRun(
            job_id=str(run["job_id"]),
            bucket=str(run["bucket"]),
            prefix=str(run["prefix"]),
            source=run["source"],
            limit=run["object_limit"],
            replace_existing=bool(run["replace_existing"]),
            status=str(run["status"]),
            submitted_at=run["submitted_at"],
            started_at=run["started_at"],
            completed_at=run["completed_at"],
            error=run["error"],
            report=report,
            work_items_total=int(summary["total"]),
            work_items_completed=int(summary["completed"]),
            work_items_retryable=int(summary["retryable"]),
            work_items_permanent_failed=int(summary["permanent_failed"]),
        )

    def mark_run_running(self, job_id: str) -> None:
        self._execute(
            """
            UPDATE ingestion_runs
            SET status = 'running', started_at = COALESCE(started_at, NOW()), error = NULL
            WHERE job_id = %s
            """,
            (UUID(job_id),),
        )

    def complete_run(self, job_id: str) -> None:
        self._execute(
            """
            UPDATE ingestion_runs
            SET status = 'completed', completed_at = NOW(), error = NULL
            WHERE job_id = %s
            """,
            (UUID(job_id),),
        )

    def fail_run(self, job_id: str, error: str) -> None:
        self._execute(
            """
            UPDATE ingestion_runs
            SET status = 'failed', completed_at = NOW(), error = %s
            WHERE job_id = %s
            """,
            (error, UUID(job_id)),
        )

    def resumable_runs(self) -> list[DurableIngestionRun]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT job_id FROM ingestion_runs WHERE status IN ('queued', 'running')"
            )
            job_ids = [str(row[0]) for row in cursor.fetchall()]
        return [run for job_id in job_ids if (run := self.get_run(job_id)) is not None]

    def add_work_item(
        self,
        job_id: str,
        bucket: str,
        key: str,
        source: str | None,
        etag: str | None,
        size_bytes: int | None,
    ) -> None:
        self._execute(
            """
            INSERT INTO ingestion_work_items (
                job_id, bucket, object_key, source, etag, size_bytes, status
            )
            VALUES (%s, %s, %s, %s, %s, %s, 'pending')
            ON CONFLICT (job_id, bucket, object_key) DO NOTHING
            """,
            (UUID(job_id), bucket, key, source, etag, size_bytes),
        )

    def claim_next_item(self, job_id: str, worker_id: UUID) -> IngestionWorkItem | None:
        with self._connect() as connection, connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                WITH next_item AS (
                    SELECT item_id
                    FROM ingestion_work_items
                    WHERE job_id = %s
                      AND (
                        status IN ('pending', 'retryable_failure')
                        OR (status = 'processing' AND lease_expires_at < NOW())
                      )
                    ORDER BY item_id
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                )
                UPDATE ingestion_work_items AS item
                SET status = 'processing',
                    attempts = item.attempts + 1,
                    lease_owner = %s,
                    lease_expires_at = NOW() + (%s * INTERVAL '1 second'),
                    started_at = COALESCE(item.started_at, NOW()),
                    error_type = NULL,
                    error = NULL
                FROM next_item
                WHERE item.item_id = next_item.item_id
                RETURNING item.*
                """,
                (UUID(job_id), worker_id, self.lease_seconds),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return IngestionWorkItem(
            item_id=int(row["item_id"]),
            job_id=str(row["job_id"]),
            bucket=str(row["bucket"]),
            key=str(row["object_key"]),
            source=row["source"],
            etag=row["etag"],
            size_bytes=row["size_bytes"],
            attempts=int(row["attempts"]),
        )

    def seconds_until_next_claim(self, job_id: str) -> float | None:
        """Return the remaining active lease duration, if any work is still processing."""
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT EXTRACT(EPOCH FROM MIN(lease_expires_at) - NOW())
                FROM ingestion_work_items
                WHERE job_id = %s AND status = 'processing'
                """,
                (UUID(job_id),),
            )
            value = cursor.fetchone()
        if value is None or value[0] is None:
            return None
        return max(0.0, float(value[0]))

    def complete_item(
        self, item: IngestionWorkItem, worker_id: UUID, report: IngestionReport, permanent: bool
    ) -> None:
        status = "permanent_failure" if permanent else "completed"
        values = asdict(report)
        self._execute(
            """
            UPDATE ingestion_work_items
            SET status = %s, lease_owner = NULL, lease_expires_at = NULL, completed_at = NOW(),
                documents_seen = %s, documents_ingested = %s,
                documents_skipped_empty = %s, documents_skipped_failed = %s,
                parents_created = %s, children_upserted = %s, batches_upserted = %s,
                documents_replaced = %s
            WHERE item_id = %s AND lease_owner = %s
            """,
            (
                status,
                values["documents_seen"],
                values["documents_ingested"],
                values["documents_skipped_empty"],
                values["documents_skipped_failed"],
                values["parents_created"],
                values["children_upserted"],
                values["batches_upserted"],
                values["documents_replaced"],
                item.item_id,
                worker_id,
            ),
        )

    def retry_item(self, item: IngestionWorkItem, worker_id: UUID, error: Exception) -> None:
        self._execute(
            """
            UPDATE ingestion_work_items
            SET status = 'retryable_failure', lease_owner = NULL, lease_expires_at = NULL,
                error_type = %s, error = %s
            WHERE item_id = %s AND lease_owner = %s
            """,
            (type(error).__name__, str(error), item.item_id, worker_id),
        )

    def _execute(self, statement: str, parameters: tuple[Any, ...]) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(statement, parameters)

    def _connect(self) -> psycopg.Connection[Any]:
        return psycopg.connect(self.database_url)
