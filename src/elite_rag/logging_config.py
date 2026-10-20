from __future__ import annotations

import logging
import sys
from pathlib import Path


def configure_logging(log_file: str, level: str) -> None:
    """Send application logs to stdout and an append-only file."""
    logger = logging.getLogger("elite_rag")
    if getattr(logger, "_elite_rag_configured", False):
        return

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(path, encoding="utf-8", delay=True)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.setLevel(numeric_level)
    logger.propagate = False
    logger.__dict__["_elite_rag_configured"] = True
