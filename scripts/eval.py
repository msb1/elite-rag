#!/usr/bin/env python3
"""Evaluate benchmark answers with faithfulness, relevancy, recall, precision, and correctness."""

from __future__ import annotations

import argparse

from elite_rag.config import get_settings
from elite_rag.eval import run_evaluation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    return run_evaluation(
        get_settings(), input_path=args.input, output_path=args.output, limit=args.limit
    )


if __name__ == "__main__":
    raise SystemExit(main())
