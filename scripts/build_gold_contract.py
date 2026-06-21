#!/usr/bin/env python
"""Build the JobMarketGold v0 contract from silver jobs (thin CLI wrapper)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as `python scripts/build_gold_contract.py` without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jobsrec.ingest.gold_contract import build_gold_contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--silver", default="data/silver/jobs.parquet")
    parser.add_argument("--out", default="data/gold_contract/job_market_gold_v0")
    parser.add_argument("--max-rows", type=int, default=500)
    args = parser.parse_args()

    manifest = build_gold_contract(args.silver, args.out, max_rows=args.max_rows)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
