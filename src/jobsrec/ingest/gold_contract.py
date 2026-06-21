"""
JobMarketGold v0 producer — builds a RAG-ready contract from silver jobs.

Reads ``data/silver/jobs.parquet`` and emits the four-file contract under
``data/gold_contract/job_market_gold_v0/``:

* ``retrieval_corpus.parquet``      — one row per sampled posting
* ``skill_share_by_period.parquet`` — skill prevalence per period (low-confidence)
* ``role_families.json``            — deterministic keyword grouping of titles
* ``dataset_manifest.json``         — provenance + artifact list

This is a sibling of ``populares.py`` (different source/schema); it does not
share that code path. Temporal coverage in silver is thin, so trend/period and
role-family signals are marked ``confidence="low"``.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

SCHEMA_VERSION = "0.0.1"
SOURCE_NAME = "linkedin_jobs_2024"  # ponytail: best-guess label, trivially editable

RETRIEVAL_CORPUS_COLUMNS = [
    "chunk_id",
    "doc_id",
    "title",
    "text",
    "source_name",
    "url",
    "section_type",
    "metadata_json",
]

SKILL_SHARE_COLUMNS = [
    "period",
    "skill_name",
    "skill_share",
    "posting_count",
    "evidence_count",
    "confidence",
]

# ponytail: naive substring rules over titles; upgrade to embeddings/clustering
# if recall matters. First match wins; postings matching none are unassigned.
ROLE_FAMILY_RULES: dict[str, dict[str, Any]] = {
    "civil_engineering": {
        "label": "Civil Engineering",
        "keywords": ["civil", "structural", "geotechnical"],
    },
    "construction_project_management": {
        "label": "Construction & Project Management",
        "keywords": ["construction", "project manager", "site", "foreman"],
    },
    "transportation_planning": {
        "label": "Transportation Planning",
        "keywords": ["transportation", "transit", "traffic", "highway"],
    },
    "water_environmental_engineering": {
        "label": "Water & Environmental Engineering",
        "keywords": ["water", "environmental", "wastewater", "hydrology"],
    },
    "data_analytics_engineering": {
        "label": "Data & Analytics Engineering",
        "keywords": ["data", "analytics", "analyst", "machine learning", "etl"],
    },
}


def _clean_str(value: Any) -> str:
    """Coerce a possibly-missing cell to a stripped string ('' for NaN/None)."""
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _split_skills(skills_text: Any) -> list[str]:
    raw = _clean_str(skills_text)
    if not raw:
        return []
    return [s.strip() for s in raw.split(",") if s.strip()]


def _period_for(listed_time: Any) -> str:
    """Month string ``YYYY-MM`` from a Unix-ms timestamp, else ``unknown``."""
    if listed_time is None or (isinstance(listed_time, float) and pd.isna(listed_time)):
        return "unknown"
    try:
        ts = pd.to_datetime(int(float(listed_time)), unit="ms", utc=True)
    except (ValueError, TypeError, OverflowError):
        return "unknown"
    if pd.isna(ts):
        return "unknown"
    return ts.strftime("%Y-%m")


def _row_text(row: pd.Series) -> str:
    """Prefer the precomputed job_card_text; fall back to the builder."""
    precomputed = _clean_str(row.get("job_card_text"))
    if precomputed:
        return precomputed
    from jobsrec.text.job_card import build_job_card_text

    return build_job_card_text(
        title=_clean_str(row.get("title")),
        description=_clean_str(row.get("description")),
        experience=_clean_str(row.get("formatted_experience_level")),
        work_type=_clean_str(row.get("formatted_work_type")),
        location=_clean_str(row.get("location")),
        skills_text=_clean_str(row.get("skills_text")),
    )


def _build_retrieval_corpus(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        doc_id = _clean_str(row.get("job_id")) or str(row.name)
        url = _clean_str(row.get("application_url"))
        metadata = {
            "location": _clean_str(row.get("location")),
            "work_type": _clean_str(row.get("formatted_work_type")),
            "experience": _clean_str(row.get("formatted_experience_level")),
            "skills_text": _clean_str(row.get("skills_text")),
            "period": _period_for(row.get("listed_time")),
        }
        rows.append(
            {
                "chunk_id": f"{doc_id}-0",
                "doc_id": doc_id,
                "title": _clean_str(row.get("title")),
                "text": _row_text(row),
                "source_name": SOURCE_NAME,
                "url": url or None,
                "section_type": "job_posting",
                "metadata_json": json.dumps(metadata, ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows, columns=RETRIEVAL_CORPUS_COLUMNS)


def _build_skill_share(df: pd.DataFrame) -> pd.DataFrame:
    # postings per period, and per-(period, skill) evidence counts
    postings_per_period: Counter[str] = Counter()
    evidence: Counter[tuple[str, str]] = Counter()
    for _, row in df.iterrows():
        period = _period_for(row.get("listed_time"))
        postings_per_period[period] += 1
        for skill in set(_split_skills(row.get("skills_text"))):
            evidence[(period, skill)] += 1

    rows: list[dict[str, Any]] = []
    for (period, skill), evidence_count in sorted(evidence.items()):
        posting_count = postings_per_period[period]
        rows.append(
            {
                "period": period,
                "skill_name": skill,
                "skill_share": evidence_count / posting_count if posting_count else 0.0,
                "posting_count": posting_count,
                "evidence_count": evidence_count,
                "confidence": "low",
            }
        )
    return pd.DataFrame(rows, columns=SKILL_SHARE_COLUMNS)


def _build_role_families(df: pd.DataFrame) -> dict[str, Any]:
    # accumulate per family
    acc: dict[str, dict[str, Any]] = {
        fam_id: {
            "titles": [],
            "skills": Counter(),
            "locations": [],
            "count": 0,
        }
        for fam_id in ROLE_FAMILY_RULES
    }

    for _, row in df.iterrows():
        title = _clean_str(row.get("title"))
        title_lc = title.lower()
        for fam_id, rule in ROLE_FAMILY_RULES.items():
            if any(kw in title_lc for kw in rule["keywords"]):
                bucket = acc[fam_id]
                bucket["count"] += 1
                if title and title not in bucket["titles"]:
                    bucket["titles"].append(title)
                for skill in _split_skills(row.get("skills_text")):
                    bucket["skills"][skill] += 1
                loc = _clean_str(row.get("location"))
                if loc and loc not in bucket["locations"]:
                    bucket["locations"].append(loc)
                break  # first match wins

    families = []
    for fam_id, rule in ROLE_FAMILY_RULES.items():
        bucket = acc[fam_id]
        families.append(
            {
                "role_family_id": fam_id,
                "label": rule["label"],
                "representative_titles": bucket["titles"][:5],
                "skills": [s for s, _ in bucket["skills"].most_common(10)],
                "locations": bucket["locations"][:10],
                "posting_count": bucket["count"],
                "confidence": "low",
            }
        )
    return {"schema_version": SCHEMA_VERSION, "role_families": families}


def build_gold_contract(
    silver_path: str | Path,
    out_dir: str | Path,
    max_rows: int = 500,
) -> dict[str, Any]:
    """Build the JobMarketGold v0 contract; returns the manifest dict."""
    df = pd.read_parquet(silver_path)
    if max_rows is not None and max_rows >= 0:
        df = df.head(max_rows)  # deterministic sample, no shuffle

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    retrieval_corpus = _build_retrieval_corpus(df)
    skill_share = _build_skill_share(df)
    role_families = _build_role_families(df)

    retrieval_corpus.to_parquet(out / "retrieval_corpus.parquet", index=False)
    skill_share.to_parquet(out / "skill_share_by_period.parquet", index=False)
    (out / "role_families.json").write_text(
        json.dumps(role_families, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    created_at = datetime.now(timezone.utc).isoformat()
    artifacts = [
        "retrieval_corpus.parquet",
        "skill_share_by_period.parquet",
        "role_families.json",
        "dataset_manifest.json",
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": f"job-market-gold-v0-{created_at}",
        "source": str(silver_path),
        "created_at": created_at,
        "row_count": int(len(retrieval_corpus)),
        # ponytail: thin temporal coverage -> pinned low; upgrade path is
        # compute_temporal_audit() in jobsrec/trends/temporal.py.
        "temporal_confidence": "low",
        "build_mode": "mvp",
        "artifacts": artifacts,
    }
    (out / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest
