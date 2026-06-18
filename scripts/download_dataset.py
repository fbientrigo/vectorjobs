#!/usr/bin/env python3
"""Download the LinkedIn job-postings dataset from Kaggle (cloud-ready PoC).

Dataset: ``arshkon/linkedin-job-postings`` -- the same source the ``jobsrec``
pipeline already references in ``docs/specs/01_data_contract.md`` and
``notebooks/cluster_skill_timelines_colab.ipynb``.

The script is designed to run unattended in a cloud / CI environment:

* Credentials are read from the environment first (no interactive upload):
    - ``KAGGLE_USERNAME`` + ``KAGGLE_KEY``  (classic API token), or
    - ``KAGGLE_API_TOKEN``                  (newer single-token form), or
    - an existing ``~/.kaggle/kaggle.json`` file.
  ``kaggle.json`` is git-ignored, so secrets are never committed.
* The download is idempotent (it is skipped when the data is already present)
  and retries on transient network failures with exponential backoff.
* ``--sample N`` derives a small, foreign-key-consistent subset (mirroring the
  layout of ``data/sample/``) so the proof of concept stays lightweight and
  never needs the full ~500 MB drop committed to git.

Examples
--------
    # Full download + a 100-row sample (needs Kaggle credentials):
    python scripts/download_dataset.py --sample 100

    # Only (re)build the small sample from an already-downloaded raw drop:
    python scripts/download_dataset.py --skip-download --sample 100

    # Validate the sampling logic against the committed fixture (no network):
    python scripts/download_dataset.py --skip-download \
        --source data/sample --sample 25 --sample-dir /tmp/sample_check
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

DATASET = "arshkon/linkedin-job-postings"

# Expected layout shared by the Kaggle drop and ``data/sample/``.
POSTINGS_FILE = "postings.csv"
COMPANY_FILES = {
    "companies/companies.csv": "company_id",
    "companies/company_industries.csv": "company_id",
    "companies/company_specialities.csv": "company_id",
    "companies/employee_counts.csv": "company_id",
}
JOB_FILES = {
    "jobs/benefits.csv": "job_id",
    "jobs/job_industries.csv": "job_id",
    "jobs/job_skills.csv": "job_id",
    "jobs/salaries.csv": "job_id",
}
# Small lookup tables copied verbatim into the sample.
MAPPING_FILES = ["mappings/industries.csv", "mappings/skills.csv"]


# --------------------------------------------------------------------------- #
# Credentials
# --------------------------------------------------------------------------- #
def ensure_credentials() -> None:
    """Make Kaggle credentials available to the CLI, or raise with guidance.

    Supports env-var based auth (cloud-friendly) by materialising a
    ``~/.kaggle/kaggle.json`` when one is not already present. Never prints the
    secret values themselves.
    """
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        return

    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY") or os.environ.get("KAGGLE_API_TOKEN")
    if username and key:
        kaggle_json.parent.mkdir(parents=True, exist_ok=True)
        kaggle_json.write_text(json.dumps({"username": username, "key": key}))
        kaggle_json.chmod(0o600)
        print(f"Wrote Kaggle credentials from environment to {kaggle_json}")
        return

    raise SystemExit(
        "Missing Kaggle credentials. Set them before running in the cloud:\n"
        "  export KAGGLE_USERNAME=<your-username>\n"
        "  export KAGGLE_KEY=<your-api-key>\n"
        "or place a token file at ~/.kaggle/kaggle.json "
        "(generated at https://www.kaggle.com/settings/api).\n"
        "Tip: use --skip-download to (re)build the small sample from data "
        "already present, which needs no credentials."
    )


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def download(raw_dir: Path, dataset: str, force: bool, retries: int = 4) -> None:
    """Download and unzip ``dataset`` into ``raw_dir`` (idempotent + retries)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    if (raw_dir / POSTINGS_FILE).exists() and not force:
        print(f"Raw data already present at {raw_dir} -- skipping download "
              "(use --force to re-download).")
        return

    ensure_credentials()
    cmd = ["kaggle", "datasets", "download", "-d", dataset,
           "-p", str(raw_dir), "--unzip"]
    delay = 2
    for attempt in range(1, retries + 1):
        print(f"[attempt {attempt}/{retries}] {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode == 0:
            return
        if attempt == retries:
            raise SystemExit(
                f"kaggle download failed after {retries} attempts "
                f"(exit code {result.returncode})."
            )
        print(f"Download failed; retrying in {delay}s ...")
        time.sleep(delay)
        delay *= 2


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def _count_rows(path: Path) -> int:
    # Count data rows without loading the whole file into memory.
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def verify(root: Path) -> bool:
    """Check expected files exist and print a row-count summary."""
    print(f"\nVerifying dataset under {root}")
    ok = True
    postings = root / POSTINGS_FILE
    if postings.exists():
        print(f"  {POSTINGS_FILE:<40} {_count_rows(postings):>9,} rows")
    else:
        print(f"  {POSTINGS_FILE:<40} MISSING")
        ok = False

    for rel in list(COMPANY_FILES) + list(JOB_FILES) + MAPPING_FILES:
        path = root / rel
        if path.exists():
            print(f"  {rel:<40} {_count_rows(path):>9,} rows")
        else:
            print(f"  {rel:<40} (absent)")
    return ok


# --------------------------------------------------------------------------- #
# Sampling (foreign-key consistent)
# --------------------------------------------------------------------------- #
def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, low_memory=False)


def _filter_by(path: Path, column: str, keep: set, out_path: Path) -> None:
    frame = _read_csv(path)
    if frame is None:
        return
    if column in frame.columns:
        frame = frame[frame[column].isin(keep)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    print(f"  {out_path.relative_to(out_path.parents[1])}: {len(frame):,} rows")


def build_sample(source: Path, sample_dir: Path, n: int) -> None:
    """Write an N-posting, FK-consistent subset mirroring data/sample/."""
    src_postings = source / POSTINGS_FILE
    if not src_postings.exists():
        raise SystemExit(f"Cannot sample: {src_postings} not found.")

    print(f"\nBuilding {n}-row sample from {source} -> {sample_dir}")
    # nrows keeps this cheap even when postings.csv is ~500 MB.
    postings = pd.read_csv(src_postings, nrows=n, low_memory=False)
    job_ids = set(postings["job_id"])
    company_ids = (
        set(postings["company_id"].dropna().astype("int64"))
        if "company_id" in postings.columns
        else set()
    )

    sample_dir.mkdir(parents=True, exist_ok=True)
    postings.to_csv(sample_dir / POSTINGS_FILE, index=False)
    print(f"  {POSTINGS_FILE}: {len(postings):,} rows "
          f"({len(company_ids):,} companies)")

    for rel in JOB_FILES:
        _filter_by(source / rel, "job_id", job_ids, sample_dir / rel)
    for rel in COMPANY_FILES:
        _filter_by(source / rel, "company_id", company_ids, sample_dir / rel)
    for rel in MAPPING_FILES:
        src = source / rel
        if src.exists():
            dst = sample_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
            print(f"  {rel}: copied")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=DATASET,
                        help=f"Kaggle dataset slug (default: {DATASET}).")
    parser.add_argument("--out", default="data/raw/linkedin-job-postings",
                        help="Destination for the raw download.")
    parser.add_argument("--sample", type=int, default=100,
                        help="Rows for the small sample (0 disables sampling).")
    parser.add_argument("--sample-dir", default="data/sample",
                        help="Destination for the FK-consistent sample.")
    parser.add_argument("--source", default=None,
                        help="Override the sampling source (defaults to --out). "
                             "Use data/sample to self-test without a download.")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip the Kaggle download (sample/verify only).")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if raw data already exists.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    raw_dir = Path(args.out)

    if not args.skip_download:
        download(raw_dir, args.dataset, args.force)

    source = Path(args.source) if args.source else raw_dir
    verify(source)

    if args.sample and args.sample > 0:
        build_sample(source, Path(args.sample_dir), args.sample)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
