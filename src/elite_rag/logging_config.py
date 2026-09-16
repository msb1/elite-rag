from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(log_file: str, level: str) -> None:
    """Configure an append-only application log once per process."""
    logger = logging.getLogger("elite_rag")
    if getattr(logger, "_elite_rag_configured", False):
        return
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8", delay=True)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = True
    logger.__dict__["_elite_rag_configured"] = True
