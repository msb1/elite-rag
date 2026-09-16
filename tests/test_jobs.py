from __future__ import annotations

import asyncio
from collections.abc import Callable

from elite_rag.ingestion import IngestionReport
from elite_rag.jobs import IngestionJobManager


def test_background_job_records_progress_and_completion() -> None:
    async def run() -> None:
        manager = IngestionJobManager()

        async def operation(progress: Callable[[IngestionReport], None]) -> IngestionReport:
            progress(IngestionReport(documents_seen=1, documents_ingested=1))
            return IngestionReport(documents_seen=2, documents_ingested=2, children_upserted=3)

        job = manager.submit("enterprise-rag", "docs", operation)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        completed = manager.get(job.job_id)
        assert completed is not None
        assert completed.status == "completed"
        assert completed.report is not None
        assert completed.report.children_upserted == 3

    asyncio.run(run())
