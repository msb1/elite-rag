from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime

from elite_rag.ingestion import IngestionReport
from elite_rag.ingestion_ledger import DurableIngestionRun
from elite_rag.jobs import IngestionJobManager


class FakeLedger:
    def __init__(self) -> None:
        self.run = DurableIngestionRun(
            job_id="00000000-0000-0000-0000-000000000001",
            bucket="enterprise-rag",
            prefix="docs",
            source=None,
            limit=None,
            replace_existing=False,
            status="queued",
            submitted_at=datetime.now(UTC),
            started_at=None,
            completed_at=None,
            error=None,
            report=IngestionReport(),
            work_items_total=0,
            work_items_completed=0,
            work_items_retryable=0,
            work_items_permanent_failed=0,
        )

    def create_or_resume_run(self, **_: object) -> DurableIngestionRun:
        return self.run

    def get_run(self, _: str) -> DurableIngestionRun:
        return self.run

    def mark_run_running(self, _: str) -> None:
        self.run = replace(self.run, status="running", started_at=datetime.now(UTC))

    def complete_run(self, _: str) -> None:
        self.run = replace(self.run, status="completed", completed_at=datetime.now(UTC))

    def fail_run(self, _: str, error: str) -> None:
        self.run = replace(self.run, status="failed", error=error)

    def resumable_runs(self) -> list[DurableIngestionRun]:
        return []


def test_background_job_records_durable_completion() -> None:
    async def run() -> None:
        manager = IngestionJobManager(FakeLedger())  # type: ignore[arg-type]

        async def operation(_: str) -> IngestionReport:
            return IngestionReport(documents_seen=2, documents_ingested=2, children_upserted=3)

        job = await manager.submit(
            bucket="enterprise-rag",
            prefix="docs",
            source=None,
            limit=None,
            replace_existing=False,
            collection_name="elite_rag",
            pipeline_version="v1",
            operation=operation,
        )
        completed = manager.get(job.job_id)
        for _ in range(20):
            if completed is not None and completed.status == "completed":
                break
            await asyncio.sleep(0.01)
            completed = manager.get(job.job_id)
        assert completed is not None
        assert completed.status == "completed"

    asyncio.run(run())
