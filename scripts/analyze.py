#!/usr/bin/env python3
"""Summarize evaluation scores and export failure rows."""

from __future__ import annotations

import argparse

from elite_rag.analyze import run_analysis
from elite_rag.config import get_settings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input")
    args = parser.parse_args()
    run_analysis(get_settings(), input_path=args.input)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
