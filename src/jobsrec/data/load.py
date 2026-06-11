"""
CSV loaders, silver-dataset writer, and manifest utilities.

Design principles
-----------------
* Functions return structured objects; nothing is printed from core logic.
* The raw Kaggle CSVs are never mutated.
* All output paths are created on-demand.
* Optional columns are preserved when present; missing optional columns are
  silently skipped (not back-filled with empty strings at the silver layer).
* Jobs without skills are kept (skills_text = "").
* A salary join from jobs/salaries.csv is attempted; if the file is absent the
  pipeline continues without salary columns.  Duplicate salary rows per job_id
  are collapsed deterministically (max of numeric fields, first non-null of
  string fields).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from jobsrec.data.schema import (
    JOB_SKILLS_REQUIRED,
    POSTINGS_OPTIONAL,
    POSTINGS_REQUIRED,
    SALARIES_OPTIONAL,
    SILVER_REQUIRED,
    SKILLS_REQUIRED,
    assert_columns,
)
from jobsrec.text.job_card import build_job_card_text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Return-value types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SilverResult:
    """Outcome of :func:`build_silver`."""

    output_path: Path
    manifest_path: Path
    input_rows: int
    output_rows: int


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_postings(input_dir: Path) -> pd.DataFrame:
    """
    Load ``postings.csv`` from *input_dir* and validate required columns.

    Parameters
    ----------
    input_dir:
        Directory containing the raw Kaggle files.

    Returns
    -------
    pd.DataFrame
        DataFrame with at least the required postings columns.

    Raises
    ------
    FileNotFoundError
        If ``postings.csv`` does not exist under *input_dir*.
    ValueError
        If any required column is missing.
    """
    path = Path(input_dir) / "postings.csv"
    if not path.exists():
        raise FileNotFoundError(f"postings.csv not found at {path}")

    df = pd.read_csv(path, low_memory=False)
    assert_columns(df, POSTINGS_REQUIRED, source="postings.csv")
    logger.info("Loaded postings.csv: %d rows", len(df))
    return df


def load_job_skills(input_dir: Path) -> pd.DataFrame:
    """
    Load ``jobs/job_skills.csv`` and validate required columns.

    Parameters
    ----------
    input_dir:
        Directory containing the raw Kaggle files.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
    ValueError
    """
    path = Path(input_dir) / "jobs" / "job_skills.csv"
    if not path.exists():
        raise FileNotFoundError(f"job_skills.csv not found at {path}")

    df = pd.read_csv(path, low_memory=False)
    assert_columns(df, JOB_SKILLS_REQUIRED, source="job_skills.csv")
    logger.info("Loaded job_skills.csv: %d rows", len(df))
    return df


def load_skills_mapping(input_dir: Path) -> pd.DataFrame:
    """
    Load ``mappings/skills.csv`` and validate required columns.

    Parameters
    ----------
    input_dir:
        Directory containing the raw Kaggle files.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
    ValueError
    """
    path = Path(input_dir) / "mappings" / "skills.csv"
    if not path.exists():
        raise FileNotFoundError(f"skills.csv not found at {path}")

    df = pd.read_csv(path, low_memory=False)
    assert_columns(df, SKILLS_REQUIRED, source="skills.csv")
    logger.info("Loaded skills.csv: %d rows", len(df))
    return df


def load_salaries(input_dir: Path) -> pd.DataFrame | None:
    """
    Load ``jobs/salaries.csv`` if it exists.

    Returns ``None`` (without raising) when the file is absent, so the
    pipeline can continue without salary data.

    Parameters
    ----------
    input_dir:
        Directory containing the raw Kaggle files.

    Returns
    -------
    pd.DataFrame or None
    """
    path = Path(input_dir) / "jobs" / "salaries.csv"
    if not path.exists():
        logger.info("jobs/salaries.csv not found — skipping salary join")
        return None

    df = pd.read_csv(path, low_memory=False)
    if "job_id" not in df.columns:
        logger.warning(
            "salaries.csv is missing 'job_id' column — skipping salary join"
        )
        return None

    logger.info("Loaded salaries.csv: %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# Skills join helper
# ---------------------------------------------------------------------------

def build_skills_text(
    job_skills: pd.DataFrame,
    skills_mapping: pd.DataFrame,
) -> pd.Series:
    """
    Join job_skills with the abbreviation→name mapping and produce a
    comma-separated ``skills_text`` Series indexed by ``job_id``.

    Parameters
    ----------
    job_skills:
        DataFrame with columns ``job_id``, ``skill_abr``.
    skills_mapping:
        DataFrame with columns ``skill_abr``, ``skill_name``.

    Returns
    -------
    pd.Series
        Index = ``job_id``, values = comma-joined skill names (or ``""``).
    """
    joined = job_skills.merge(skills_mapping, on="skill_abr", how="left")
    # Deduplicate in case of repeated skill_abr per job
    joined = joined.drop_duplicates(subset=["job_id", "skill_abr"])
    skills_series = (
        joined.groupby("job_id")["skill_name"]
        .apply(lambda names: ", ".join(n for n in names if pd.notna(n)))
        .rename("skills_text")
    )
    return skills_series


# ---------------------------------------------------------------------------
# Salary aggregation helper
# ---------------------------------------------------------------------------

def aggregate_salaries(salaries_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse multiple salary rows per ``job_id`` into one row.

    Aggregation rules (deterministic):
    * Numeric columns: max of non-null values (conservative upper bound).
    * String columns: first non-null, non-empty value.

    Only columns present in ``SALARIES_OPTIONAL`` are kept (plus ``job_id``).

    Parameters
    ----------
    salaries_df:
        Raw salaries DataFrame (may have multiple rows per job_id).

    Returns
    -------
    pd.DataFrame
        One row per ``job_id`` with the salary columns that were present.
    """
    numeric_cols = [
        c
        for c in ("min_salary", "max_salary", "med_salary")
        if c in salaries_df.columns
    ]
    string_cols = [
        c
        for c in ("pay_period", "currency", "compensation_type")
        if c in salaries_df.columns
    ]

    agg: dict[str, Any] = {}
    for col in numeric_cols:
        agg[col] = "max"
    for col in string_cols:
        # first non-null, non-empty value
        agg[col] = lambda s: next(
            (v for v in s if pd.notna(v) and str(v).strip() != ""), None
        )

    if not agg:
        # Nothing to aggregate; return unique job_ids only
        return salaries_df[["job_id"]].drop_duplicates()

    agg_df = salaries_df.groupby("job_id", as_index=False).agg(agg)
    return agg_df


# ---------------------------------------------------------------------------
# Silver writer
# ---------------------------------------------------------------------------

def build_silver(
    input_dir: Path | str,
    output_dir: Path | str,
    config: dict[str, Any] | None = None,
) -> SilverResult:
    """
    Load raw CSVs → join → build job_card_text → write silver Parquet.

    Behaviour
    ---------
    * All optional postings columns listed in :data:`POSTINGS_OPTIONAL` that
      are present in the file are carried through unchanged (temporal columns
      stay as raw numeric ms-timestamp values; string columns are stripped).
    * ``jobs/salaries.csv`` is joined on ``job_id`` (left join) when available.
      Multiple salary rows per job are aggregated deterministically before the
      join so that no job row is duplicated.
    * Jobs with no matching skill rows are kept (``skills_text = ""``).
    * The manifest records which optional columns and tables were preserved.

    Parameters
    ----------
    input_dir:
        Directory containing ``postings.csv``, ``jobs/job_skills.csv``,
        ``mappings/skills.csv``, and optionally ``jobs/salaries.csv``.
    output_dir:
        Destination directory for ``jobs.parquet`` and ``manifest.json``.
    config:
        Optional config dict (currently used only for manifest logging).

    Returns
    -------
    SilverResult
        Paths and row counts for the generated outputs.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load ---
    postings = load_postings(input_dir)
    job_skills = load_job_skills(input_dir)
    skills_mapping = load_skills_mapping(input_dir)
    salaries_raw = load_salaries(input_dir)

    input_rows = len(postings)

    # --- Normalise required columns ---
    postings["title"] = postings["title"].fillna("").astype(str).str.strip()
    postings["description"] = (
        postings["description"].fillna("").astype(str).str.strip()
    )
    postings["job_id"] = postings["job_id"].astype("int64")

    # --- Preserve optional postings columns that actually exist ---
    #: Temporal columns are left as numeric (ms-timestamps); string columns
    #: are stripped of whitespace.
    _TEMPORAL_OPTIONAL = frozenset(
        {"listed_time", "original_listed_time", "expiry", "closed_time"}
    )
    preserved_optional_cols: list[str] = []
    for col in POSTINGS_OPTIONAL:
        if col in postings.columns:
            preserved_optional_cols.append(col)
            if col not in _TEMPORAL_OPTIONAL:
                postings[col] = (
                    postings[col].fillna("").astype(str).str.strip()
                )
            # else: keep raw numeric / NaN for temporal columns

    logger.info(
        "Preserved optional postings columns (%d): %s",
        len(preserved_optional_cols),
        preserved_optional_cols,
    )

    # --- Join skills (left join → jobs without skills kept) ---
    skills_text_series = build_skills_text(job_skills, skills_mapping)
    postings = postings.merge(skills_text_series, on="job_id", how="left")
    postings["skills_text"] = postings["skills_text"].fillna("")

    # --- Salary join ---
    joined_optional_tables: list[str] = []
    salary_cols_added: list[str] = []
    if salaries_raw is not None:
        salary_agg = aggregate_salaries(salaries_raw)
        salary_cols_present = [
            c for c in SALARIES_OPTIONAL if c in salary_agg.columns
        ]
        if salary_cols_present:
            salary_agg = salary_agg[["job_id"] + salary_cols_present].copy()
            # Cast job_id to int64 for join
            salary_agg["job_id"] = salary_agg["job_id"].astype("int64")
            
            drop_cols = [c for c in salary_cols_present if c in postings.columns]
            if drop_cols:
                postings = postings.drop(columns=drop_cols)

            before = len(postings)
            postings = postings.merge(salary_agg, on="job_id", how="left")
            after = len(postings)
            if after != before:
                logger.error(
                    "Salary join duplicated rows! before=%d after=%d",
                    before,
                    after,
                )
                raise RuntimeError(
                    f"Salary join introduced duplicate rows "
                    f"(before={before}, after={after}). "
                    "Investigate aggregate_salaries()."
                )
            joined_optional_tables.append("jobs/salaries.csv")
            salary_cols_added = salary_cols_present
            logger.info(
                "Salary join added columns: %s (%d rows unchanged)",
                salary_cols_added,
                after,
            )

    # --- Build job_card_text ---
    postings["job_card_text"] = postings.apply(
        lambda row: build_job_card_text(
            title=row["title"],
            description=row["description"],
            experience=str(row.get("formatted_experience_level", "") or ""),
            work_type=str(row.get("formatted_work_type", "") or ""),
            location=str(row.get("location", "") or ""),
            skills_text=str(row.get("skills_text", "") or ""),
        ),
        axis=1,
    )

    # --- Select silver columns (required + optional present) ---
    keep_cols: list[str] = list(SILVER_REQUIRED)
    for col in preserved_optional_cols:
        if col not in keep_cols:
            keep_cols.append(col)
    for col in salary_cols_added:
        if col not in keep_cols:
            keep_cols.append(col)

    silver = postings[[c for c in keep_cols if c in postings.columns]].copy()

    # Drop rows where job_card_text ended up empty (defensive)
    silver = silver[silver["job_card_text"].str.strip() != ""]

    output_rows = len(silver)

    # --- Write Parquet ---
    output_path = output_dir / "jobs.parquet"
    silver.to_parquet(output_path, index=False)
    logger.info("Wrote silver Parquet: %s (%d rows)", output_path, output_rows)

    # --- Write manifest ---
    manifest = _make_manifest(
        stage="build-silver",
        input_rows=input_rows,
        output_rows=output_rows,
        input_path=str(input_dir),
        output_path=str(output_path),
        config=config or {},
        preserved_optional_columns=preserved_optional_cols,
        joined_optional_tables=joined_optional_tables,
        salary_columns_added=salary_cols_added,
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Wrote manifest: %s", manifest_path)

    return SilverResult(
        output_path=output_path,
        manifest_path=manifest_path,
        input_rows=input_rows,
        output_rows=output_rows,
    )


# ---------------------------------------------------------------------------
# Manifest helper
# ---------------------------------------------------------------------------

def _make_manifest(
    stage: str,
    input_rows: int,
    output_rows: int,
    input_path: str,
    output_path: str,
    config: dict[str, Any],
    preserved_optional_columns: list[str] | None = None,
    joined_optional_tables: list[str] | None = None,
    salary_columns_added: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "input_rows": input_rows,
        "output_rows": output_rows,
        "input_path": input_path,
        "output_path": output_path,
        "config": config,
        "preserved_optional_columns": preserved_optional_columns or [],
        "joined_optional_tables": joined_optional_tables or [],
        "salary_columns_added": salary_columns_added or [],
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write *manifest* dict as pretty-printed JSON to *path*."""
    path.write_text(json.dumps(manifest, indent=2))
