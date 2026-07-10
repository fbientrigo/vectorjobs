#!/usr/bin/env python
"""Build the frozen 50-row job sample used by this experiment.

Reads data/silver/jobs.parquet (read-only), filters, and samples deterministically
with random_state=3407. Re-running this script reproduces the exact same sample.

Schema note: data/silver/jobs.parquet has no "company", "posted_date", or "source"
columns. This script maps company_name -> company, first_seen_at -> posted_date,
and fills source with the constant "linkedin" (this dataset has a single source:
the LinkedIn scraper in data/bronze/jobs.db). See sample_manifest.json for the
recorded mapping.
"""

import json
from pathlib import Path

import pandas as pd

EXPERIMENT_DIR = Path(__file__).parent
SILVER_PATH = Path("data/silver/jobs.parquet")
OUTPUT_JSONL = EXPERIMENT_DIR / "input" / "jobs_sample_50.jsonl"
OUTPUT_MANIFEST = EXPERIMENT_DIR / "input" / "sample_manifest.json"

RANDOM_STATE = 3407
SAMPLE_SIZE = 50
MIN_JOB_CARD_LEN = 300
MAX_JOB_CARD_LEN = 5000
MIN_DESCRIPTION_LEN = 200

OUTPUT_COLUMNS = [
    "job_id",
    "title",
    "company",
    "location",
    "posted_date",
    "source",
    "job_card_text",
    "description_text",
]


def main() -> None:
    if not SILVER_PATH.exists():
        raise FileNotFoundError(
            f"{SILVER_PATH} not found. Build it first with the existing pipeline "
            "(scripts/run_pipeline.py or jobsrec.data.load.build_silver)."
        )

    df = pd.read_parquet(SILVER_PATH)
    n_input_rows = len(df)

    job_card_len = df["job_card_text"].fillna("").str.len()
    description_len = df["description_text"].fillna("").str.len()
    mask = (
        job_card_len.between(MIN_JOB_CARD_LEN, MAX_JOB_CARD_LEN)
        & (description_len >= MIN_DESCRIPTION_LEN)
    )
    filtered = df[mask]
    n_filtered_rows = len(filtered)

    if n_filtered_rows < SAMPLE_SIZE:
        raise ValueError(
            f"Only {n_filtered_rows} rows pass the filter, need at least {SAMPLE_SIZE}."
        )

    sample = filtered.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).copy()
    sample["company"] = sample["company_name"]
    sample["posted_date"] = sample["first_seen_at"]
    sample["source"] = "linkedin"
    sample = sample[OUTPUT_COLUMNS]

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        for _, row in sample.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False, default=str) + "\n")

    manifest = {
        "silver_path": str(SILVER_PATH),
        "random_state": RANDOM_STATE,
        "sample_size": SAMPLE_SIZE,
        "n_input_rows": n_input_rows,
        "n_filtered_rows": n_filtered_rows,
        "filters": {
            "job_card_text_len_min": MIN_JOB_CARD_LEN,
            "job_card_text_len_max": MAX_JOB_CARD_LEN,
            "description_text_len_min": MIN_DESCRIPTION_LEN,
        },
        "column_mapping": {
            "company": "company_name",
            "posted_date": "first_seen_at",
            "source": "constant:linkedin (single-source dataset, no source column in silver schema)",
        },
        "output_path": str(OUTPUT_JSONL),
        "job_ids": sample["job_id"].tolist(),
    }
    OUTPUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {SAMPLE_SIZE} rows to {OUTPUT_JSONL}")
    print(f"Wrote manifest to {OUTPUT_MANIFEST}")


if __name__ == "__main__":
    main()
