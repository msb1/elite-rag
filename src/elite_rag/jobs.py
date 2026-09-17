from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from threading import Lock

from elite_rag.ingestion import IngestionReport
from elite_rag.ingestion_ledger import DurableIngestionRun, IngestionLedger

LOGGER = logging.getLogger(__name__)

RunOperation = Callable[[str], Awaitable[IngestionReport]]


class IngestionJobManager:
    """Durably tracked RustFS-prefix ingestion jobs backed by PostgreSQL."""

    def __init__(self, ledger: IngestionLedger) -> None:
        self.ledger = ledger
        self._active_job_ids: set[str] = set()
        self._lock = Lock()

    async def submit(
        self,
        *,
        bucket: str,
        prefix: str,
        source: str | None,
        limit: int | None,
        replace_existing: bool,
        collection_name: str,
        pipeline_version: str,
        operation: RunOperation,
    ) -> DurableIngestionRun:
        run = await asyncio.to_thread(
            self.ledger.create_or_resume_run,
            bucket=bucket,
            prefix=prefix,
            source=source,
            limit=limit,
            replace_existing=replace_existing,
            collection_name=collection_name,
            pipeline_version=pipeline_version,
        )
        self._start(run.job_id, operation)
        LOGGER.info(
            "ingestion_job_submitted job_id=%s bucket=%s prefix=%s resumed=%s",
            run.job_id,
            bucket,
            prefix,
            run.work_items_total > 0,
        )
        return run

    def get(self, job_id: str) -> DurableIngestionRun | None:
        return self.ledger.get_run(job_id)

    def resume_unfinished(self, operation: RunOperation) -> int:
        runs = self.ledger.resumable_runs()
        for run in runs:
            self._start(run.job_id, operation)
        return len(runs)

    def _start(self, job_id: str, operation: RunOperation) -> None:
        with self._lock:
            if job_id in self._active_job_ids:
                return
            self._active_job_ids.add(job_id)
        asyncio.create_task(self._run(job_id, operation), name=f"ingest-{job_id}")

    async def _run(self, job_id: str, operation: RunOperation) -> None:
        await asyncio.to_thread(self.ledger.mark_run_running, job_id)
        LOGGER.info("ingestion_job_started job_id=%s", job_id)
        try:
            report = await operation(job_id)
        except Exception as exc:
            LOGGER.exception(
                "ingestion_job_failed job_id=%s error_type=%s", job_id, type(exc).__name__
            )
            await asyncio.to_thread(self.ledger.fail_run, job_id, f"{type(exc).__name__}: {exc}")
            return
        finally:
            with self._lock:
                self._active_job_ids.discard(job_id)
        await asyncio.to_thread(self.ledger.complete_run, job_id)
        LOGGER.info("ingestion_job_completed job_id=%s report=%s", job_id, report)
