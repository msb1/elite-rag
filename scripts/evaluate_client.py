#!/usr/bin/env python3
"""Send one answer record to the Elite RAG custom evaluation API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

from elite_rag.config import get_settings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "record", type=Path, help="JSON file with question, answer, contexts, ground_truth"
    )
    settings = get_settings()
    parser.add_argument(
        "--server", default=f"http://{settings.server_host}:{settings.server_port}"
    )
    parser.add_argument("--api-key", default=settings.api_key)
    args = parser.parse_args()
    payload = json.loads(args.record.read_text(encoding="utf-8"))
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    request = Request(
        f"{args.server.rstrip('/')}/v1/evaluate",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urlopen(request, timeout=900) as response:
        print(json.dumps(json.load(response), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
