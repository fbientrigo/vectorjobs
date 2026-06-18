#!/usr/bin/env python3
"""Get the LinkedIn job-postings dataset in a cloud environment (multi-source).

Dataset: ``arshkon/linkedin-job-postings`` -- the source the ``jobsrec``
pipeline already references in ``docs/specs/01_data_contract.md`` and
``notebooks/cluster_skill_timelines_colab.ipynb``.

Cloud reality
-------------
Remote sessions enforce a **network egress allowlist** *and* (for Kaggle) need
credentials.  In a fresh Claude-Code-on-the-web sandbox ``github.com`` is
reachable but ``kaggle.com`` / ``huggingface.co`` are not, so a single download
path is fragile.  This script therefore supports several sources and degrades
gracefully:

``--source-type``
    ``kaggle``  Kaggle API (needs creds + ``kaggle.com`` allowlisted).
    ``url``     Any public archive (zip/tar.gz) over HTTPS -- e.g. a GitHub
                Release asset or a team bucket on an allowlisted host. No creds.
    ``hf``      Hugging Face dataset mirror via ``huggingface_hub`` (public, no
                creds; needs ``huggingface.co`` allowlisted).
    ``sample``  No network at all: use the committed ``data/sample`` fixture as
                the source.  Always works -- ideal for a cloud PoC.

See ``scripts/README_dataset.md`` for how to wire each option (egress
allowlist, secrets, publishing a GitHub Release mirror).

Examples
--------
    # Guaranteed offline PoC (no network, no creds):
    python scripts/download_dataset.py --source-type sample --sample 100

    # Pull a team-published mirror from an allowlisted host:
    python scripts/download_dataset.py --source-type url \
        --url https://github.com/<org>/<repo>/releases/download/v1/linkedin.zip

    # Classic Kaggle (needs creds + allowlist):
    python scripts/download_dataset.py --source-type kaggle --sample 100
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

DATASET = "arshkon/linkedin-job-postings"
HF_MIRROR = "xanderios/linkedin-job-postings"
USER_AGENT = "vectorjobs-downloader/1.0"

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
MAPPING_FILES = ["mappings/industries.csv", "mappings/skills.csv"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def find_dataset_root(base: Path) -> Path:
    """Return the directory that directly contains postings.csv under ``base``."""
    if (base / POSTINGS_FILE).exists():
        return base
    for candidate in base.rglob(POSTINGS_FILE):
        return candidate.parent
    return base


def _retry(action, retries: int = 4):
    delay = 2
    for attempt in range(1, retries + 1):
        try:
            return action()
        except Exception as exc:  # noqa: BLE001 - report and back off
            if attempt == retries:
                raise
            print(f"  attempt {attempt}/{retries} failed ({exc}); "
                  f"retrying in {delay}s ...")
            time.sleep(delay)
            delay *= 2


# --------------------------------------------------------------------------- #
# Source: Kaggle
# --------------------------------------------------------------------------- #
def ensure_kaggle_credentials() -> None:
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
        "Missing Kaggle credentials. In the cloud set them as secrets:\n"
        "  export KAGGLE_USERNAME=<user> ; export KAGGLE_KEY=<key>\n"
        "and make sure kaggle.com is on the egress allowlist. If Kaggle is not "
        "reachable, use --source-type url (GitHub Release mirror) or "
        "--source-type sample (no network)."
    )


def download_kaggle(raw_dir: Path, dataset: str) -> None:
    ensure_kaggle_credentials()
    cmd = ["kaggle", "datasets", "download", "-d", dataset,
           "-p", str(raw_dir), "--unzip"]

    def run() -> None:
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise RuntimeError(f"kaggle exit code {result.returncode}")

    print(f"[kaggle] {' '.join(cmd)}")
    _retry(run)


# --------------------------------------------------------------------------- #
# Source: generic public URL (zip / tar.gz)
# --------------------------------------------------------------------------- #
def download_url(raw_dir: Path, url: str) -> None:
    archive = raw_dir / Path(url.split("?")[0]).name
    archive.parent.mkdir(parents=True, exist_ok=True)

    def fetch() -> None:
        print(f"[url] GET {url}")
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=60) as response, \
                archive.open("wb") as out:
            shutil.copyfileobj(response, out)

    _retry(fetch)
    print(f"[url] extracting {archive.name}")
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(raw_dir)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            tf.extractall(raw_dir)
    else:
        raise SystemExit(f"Unsupported archive format: {archive.name}")
    archive.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Source: Hugging Face mirror (public, no credentials)
# --------------------------------------------------------------------------- #
def download_hf(raw_dir: Path, repo_id: str) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub not installed. Run: pip install huggingface_hub\n"
            f"(import error: {exc})"
        )
    print(f"[hf] snapshot_download {repo_id} (dataset)")
    _retry(lambda: snapshot_download(
        repo_id=repo_id, repo_type="dataset", local_dir=str(raw_dir)))


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def _count_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def verify(root: Path) -> bool:
    print(f"\nVerifying dataset under {root}")
    ok = (root / POSTINGS_FILE).exists()
    for rel in [POSTINGS_FILE] + list(COMPANY_FILES) + list(JOB_FILES) + MAPPING_FILES:
        path = root / rel
        status = f"{_count_rows(path):>9,} rows" if path.exists() else "(absent)"
        print(f"  {rel:<40} {status}")
    if not ok:
        print(f"  WARNING: {POSTINGS_FILE} not found under {root}")
    return ok


# --------------------------------------------------------------------------- #
# Sampling (foreign-key consistent)
# --------------------------------------------------------------------------- #
def _read_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path, low_memory=False) if path.exists() else None


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
    src_postings = source / POSTINGS_FILE
    if not src_postings.exists():
        raise SystemExit(f"Cannot sample: {src_postings} not found.")

    print(f"\nBuilding {n}-row sample from {source} -> {sample_dir}")
    postings = pd.read_csv(src_postings, nrows=n, low_memory=False)
    job_ids = set(postings["job_id"])
    company_ids = (
        set(postings["company_id"].dropna().astype("int64"))
        if "company_id" in postings.columns else set()
    )

    sample_dir.mkdir(parents=True, exist_ok=True)
    postings.to_csv(sample_dir / POSTINGS_FILE, index=False)
    print(f"  {POSTINGS_FILE}: {len(postings):,} rows ({len(company_ids):,} companies)")

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
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-type", default="kaggle",
                        choices=["kaggle", "url", "hf", "sample"],
                        help="Where to get the dataset (default: kaggle).")
    parser.add_argument("--dataset", default=DATASET,
                        help=f"Kaggle dataset slug (default: {DATASET}).")
    parser.add_argument("--url", default=os.environ.get("DATASET_URL"),
                        help="Archive URL for --source-type url (or DATASET_URL env).")
    parser.add_argument("--hf-repo", default=HF_MIRROR,
                        help=f"HF dataset repo for --source-type hf (default: {HF_MIRROR}).")
    parser.add_argument("--out", default="data/raw/linkedin-job-postings",
                        help="Destination for the raw download.")
    parser.add_argument("--sample", type=int, default=100,
                        help="Rows for the small sample (0 disables sampling).")
    parser.add_argument("--sample-dir", default="data/sample",
                        help="Destination for the FK-consistent sample.")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if raw data already exists.")
    return parser.parse_args(argv)


def acquire(args: argparse.Namespace) -> Path:
    """Run the chosen source and return the directory holding postings.csv."""
    if args.source_type == "sample":
        # No network: the committed fixture *is* the source.
        return Path(args.sample_dir)

    raw_dir = Path(args.out)
    raw_dir.mkdir(parents=True, exist_ok=True)
    if (raw_dir / POSTINGS_FILE).exists() and not args.force:
        print(f"Raw data already present at {raw_dir} -- skipping download "
              "(use --force to re-download).")
        return find_dataset_root(raw_dir)

    if args.source_type == "kaggle":
        download_kaggle(raw_dir, args.dataset)
    elif args.source_type == "url":
        if not args.url:
            raise SystemExit("--source-type url requires --url or DATASET_URL.")
        download_url(raw_dir, args.url)
    elif args.source_type == "hf":
        download_hf(raw_dir, args.hf_repo)
    return find_dataset_root(raw_dir)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.source_type == "sample":
        # Sample-from-sample is a no-op; sample into a sibling dir if asked.
        source = Path(args.sample_dir)
        verify(source)
        if args.sample and args.sample > 0 and args.sample_dir != "data/sample":
            build_sample(Path("data/sample"), source, args.sample)
        print("\nDone (offline sample source).")
        return 0

    source = acquire(args)
    verify(source)
    if args.sample and args.sample > 0:
        build_sample(source, Path(args.sample_dir), args.sample)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
