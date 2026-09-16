from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from threading import Lock
from uuid import uuid4

from elite_rag.ingestion import IngestionReport

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class IngestionJob:
    job_id: str
    bucket: str
    prefix: str
    status: str = "queued"
    submitted_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    report: IngestionReport | None = None
    error: str | None = None

    def response(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "bucket": self.bucket,
            "prefix": self.prefix,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "report": asdict(self.report) if self.report else None,
            "error": self.error,
        }


class IngestionJobManager:
    """In-process job tracker for long-running RustFS prefix ingestion."""

    def __init__(self) -> None:
        self._jobs: dict[str, IngestionJob] = {}
        self._lock = Lock()

    def submit(
        self,
        bucket: str,
        prefix: str,
        operation: Callable[[Callable[[IngestionReport], None]], Awaitable[IngestionReport]],
    ) -> IngestionJob:
        job = IngestionJob(job_id=str(uuid4()), bucket=bucket, prefix=prefix)
        with self._lock:
            self._jobs[job.job_id] = job
        asyncio.create_task(self._run(job.job_id, operation), name=f"ingest-{job.job_id}")
        LOGGER.info(
            "ingestion_job_submitted job_id=%s bucket=%s prefix=%s", job.job_id, bucket, prefix
        )
        return job

    def get(self, job_id: str) -> IngestionJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    async def _run(
        self,
        job_id: str,
        operation: Callable[[Callable[[IngestionReport], None]], Awaitable[IngestionReport]],
    ) -> None:
        self._update(job_id, status="running", started_at=datetime.now(UTC))
        LOGGER.info("ingestion_job_started job_id=%s", job_id)
        try:
            report = await operation(lambda progress: self._update(job_id, report=progress))
        except Exception as exc:
            LOGGER.exception(
                "ingestion_job_failed job_id=%s error_type=%s", job_id, type(exc).__name__
            )
            self._update(
                job_id,
                status="failed",
                completed_at=datetime.now(UTC),
                error=f"{type(exc).__name__}: {exc}",
            )
            return
        self._update(
            job_id,
            status="completed",
            completed_at=datetime.now(UTC),
            report=report,
        )
        LOGGER.info("ingestion_job_completed job_id=%s report=%s", job_id, asdict(report))

    def _update(self, job_id: str, **changes: object) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for name, value in changes.items():
                setattr(job, name, value)
