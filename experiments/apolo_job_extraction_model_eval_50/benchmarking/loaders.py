"""Read-only loaders over experiment outputs. No network calls, no writes."""

import csv
import json
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parent.parent
OUTPUTS_DIR = EXPERIMENT_DIR / "outputs"
REPORTS_DIR = EXPERIMENT_DIR / "reports"
SAMPLE_MANIFEST_PATH = EXPERIMENT_DIR / "input" / "sample_manifest.json"
SAMPLE_PATH = EXPERIMENT_DIR / "input" / "jobs_sample_50.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def safe_model_name(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


def load_frozen_job_ids(sample_manifest_path: Path = SAMPLE_MANIFEST_PATH) -> list[str]:
    """The 50 job_ids this whole experiment is frozen to (cohort A)."""
    manifest = json.loads(sample_manifest_path.read_text(encoding="utf-8"))
    return [str(job_id) for job_id in manifest["job_ids"]]


def load_frozen_sample(sample_path: Path = SAMPLE_PATH) -> dict[str, dict]:
    return {str(row["job_id"]): row for row in read_jsonl(sample_path)}


def load_run_manifest(model: str, outputs_dir: Path = OUTPUTS_DIR) -> dict | None:
    path = outputs_dir / safe_model_name(model) / "manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def discover_models(outputs_dir: Path = OUTPUTS_DIR) -> list[str]:
    manifests = sorted(outputs_dir.glob("*/manifest.json"))
    return [json.loads(p.read_text(encoding="utf-8"))["model"] for p in manifests]


def load_all_run_manifests(outputs_dir: Path = OUTPUTS_DIR) -> dict[str, dict]:
    rows = {}
    for path in sorted(outputs_dir.glob("*/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        rows[manifest.get("display_name") or manifest["model"]] = manifest
    return rows


def load_extractions(model: str, outputs_dir: Path = OUTPUTS_DIR) -> list[dict]:
    for path in sorted(outputs_dir.glob("*/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        labels = {
            manifest.get("display_name"),
            manifest.get("model"),
            manifest.get("benchmark_model"),
            manifest.get("configuration_id"),
            manifest.get("output_slug"),
        }
        if model in labels:
            return read_jsonl(path.parent / "extractions.jsonl")
    return read_jsonl(outputs_dir / safe_model_name(model) / "extractions.jsonl")


def load_gold_vs_model(reports_dir: Path = REPORTS_DIR) -> dict[str, dict] | None:
    """Parse reports/model_vs_gold.csv if evaluate_against_gold.py has already run.

    Returns None (not an empty dict) when no gold evaluation exists yet, so
    callers can distinguish "not evaluated" from "evaluated, zero rows".
    """
    path = reports_dir / "model_vs_gold.csv"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None

    def parse(value):
        if value == "" or value is None:
            return None
        try:
            return float(value)
        except ValueError:
            return value

    return {row["model"]: {k: parse(v) for k, v in row.items() if k != "model"} for row in rows}
