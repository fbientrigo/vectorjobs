"""Build job_extraction_candidates DataFrame from silver data."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from jobsrec.extract.html_clean import extract_ordered
from jobsrec.extract.sections import detect_section
from jobsrec.extract.skills import SKILL_DICT_VERSION, match_skills

logger = logging.getLogger(__name__)

EXTRACTION_SCHEMA_VERSION = "extraction_v0.1"

CANDIDATE_COLUMNS: tuple[str, ...] = (
    "job_id",
    "candidate_index",
    "candidate_text",
    "candidate_source",
    "section_name",
    "skills_regex_raw",
    "skills_normalized",
)


def build_extraction_candidates(silver_df: pd.DataFrame) -> pd.DataFrame:
    """Return job_extraction_candidates DataFrame.

    One row per text unit (title, paragraph, or li item) per job.
    skills_regex_raw and skills_normalized are JSON-encoded list[str] columns.
    """
    rows: list[dict[str, Any]] = []

    for _, job in silver_df.iterrows():
        job_id = str(job["job_id"])
        html_text = str(job.get("description_html") or "")
        title = str(job.get("title") or "").strip()
        idx = 0

        if title:
            raw, normed = match_skills(title)
            rows.append({
                "job_id": job_id,
                "candidate_index": idx,
                "candidate_text": title,
                "candidate_source": "title",
                "section_name": "",
                "skills_regex_raw": json.dumps(raw, ensure_ascii=False),
                "skills_normalized": json.dumps(normed, ensure_ascii=False),
            })
            idx += 1

        current_section = ""
        for source, text in extract_ordered(html_text):
            detected = detect_section(text)
            if detected:
                current_section = detected
            raw, normed = match_skills(text)
            rows.append({
                "job_id": job_id,
                "candidate_index": idx,
                "candidate_text": text,
                "candidate_source": source,
                "section_name": current_section,
                "skills_regex_raw": json.dumps(raw, ensure_ascii=False),
                "skills_normalized": json.dumps(normed, ensure_ascii=False),
            })
            idx += 1

    if not rows:
        return pd.DataFrame(columns=list(CANDIDATE_COLUMNS))

    df = pd.DataFrame(rows, columns=list(CANDIDATE_COLUMNS))
    df["candidate_index"] = df["candidate_index"].astype("int32")
    return df


def build_extraction_report(
    df: pd.DataFrame,
    silver_df: pd.DataFrame,
    parse_error_count: int,
) -> dict[str, Any]:
    """Compute baseline extraction statistics."""
    jobs_processed = len(silver_df)
    candidate_rows = len(df)

    jobs_with_candidates = df["job_id"].nunique() if candidate_rows > 0 else 0
    jobs_with_candidates_pct = round(jobs_with_candidates / jobs_processed * 100, 1) if jobs_processed else 0.0

    from collections import Counter
    skill_counter: Counter[str] = Counter()
    jobs_with_skills: set[str] = set()
    for _, row in df.iterrows():
        normed = json.loads(row["skills_normalized"])
        if normed:
            jobs_with_skills.add(row["job_id"])
            for s in normed:
                skill_counter[s] += 1

    jobs_with_skills_pct = round(len(jobs_with_skills) / jobs_processed * 100, 1) if jobs_processed else 0.0

    return {
        "jobs_processed": jobs_processed,
        "candidate_rows": candidate_rows,
        "jobs_with_candidates_pct": jobs_with_candidates_pct,
        "jobs_with_skills_pct": jobs_with_skills_pct,
        "top_skills": skill_counter.most_common(20),
        "company_parse_errors": parse_error_count,
        "skill_dict_version": SKILL_DICT_VERSION,
        "extraction_schema_version": EXTRACTION_SCHEMA_VERSION,
    }


def write_extraction_manifest(
    output_dir: Path,
    report: dict[str, Any],
    silver_path: str,
    output_path: str,
) -> Path:
    """Write extraction_manifest.json and return its path."""
    manifest = {
        **report,
        "stage": "extract-candidates",
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "silver_path": silver_path,
        "output_path": output_path,
        "candidate_columns": list(CANDIDATE_COLUMNS),
        "top_skills": [{"skill": s, "count": c} for s, c in report["top_skills"]],
    }
    manifest_path = output_dir / "extraction_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path
