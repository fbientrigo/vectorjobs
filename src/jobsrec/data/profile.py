"""
Data profiling for silver Parquet datasets.

Produces a structured summary of a silver-layer jobs DataFrame including
posting counts, skill statistics, optional column availability,
date-parse diagnostics, salary column coverage, and categorical distributions
— without any network or GPU dependencies.

Design principles
-----------------
* Pure functions; nothing is printed from core logic.
* All summary values are JSON-serialisable (int, float, str, list, bool, None).
* Tolerant of missing optional columns — reports their absence rather than
  raising.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------


@dataclass
class SilverProfile:
    """Structured summary of a silver Parquet dataset."""

    # --- core counts ---
    n_postings: int
    """Total number of rows in the dataset."""

    n_unique_job_ids: int
    """Number of distinct job_id values."""

    n_missing_titles: int
    """Rows where title is empty or whitespace-only."""

    n_missing_descriptions: int
    """Rows where description is empty or whitespace-only."""

    n_jobs_without_skills: int
    """Rows where skills_text is empty or whitespace-only."""

    n_unique_skills: int
    """Number of distinct skill tokens across all postings."""

    top_20_skills: list[tuple[str, int]]
    """Top-20 (skill_name, frequency) pairs sorted descending."""

    # --- datetime ---
    datetime_columns_present: list[str]
    """Date/time column names found in the DataFrame."""

    listed_time_parse_rate: float | None
    """Fraction of listed_time values successfully parsed as ms timestamps,
    or None if listed_time is absent."""

    listed_time_n_total: int | None
    """Total listed_time values (non-null) examined, or None if absent."""

    listed_time_n_parsed: int | None
    """Count of successfully parsed listed_time values, or None if absent."""

    expiry_parse_rate: float | None
    """Fraction of expiry values successfully parsed, or None if absent."""

    expiry_n_total: int | None
    """Total expiry values (non-null) examined, or None if absent."""

    expiry_n_parsed: int | None
    """Count of successfully parsed expiry values, or None if absent."""

    closed_time_parse_rate: float | None
    """Fraction of closed_time values successfully parsed, or None if absent."""

    closed_time_n_total: int | None
    """Total closed_time values (non-null) examined, or None if absent."""

    closed_time_n_parsed: int | None
    """Count of successfully parsed closed_time values, or None if absent."""

    # --- salary ---
    salary_columns_present: list[str]
    """Salary-related column names found in the DataFrame."""

    salary_non_null_counts: dict[str, int]
    """Mapping of salary column name → count of non-null values."""

    # --- location / work-type / experience ---
    location_columns_present: list[str]
    """Location/work-type/experience column names found in the DataFrame."""

    work_type_distribution: dict[str, int] | None
    """Value counts for formatted_work_type, or None if absent."""

    experience_level_distribution: dict[str, int] | None
    """Value counts for formatted_experience_level, or None if absent."""

    location_top_20: list[tuple[str, int]] | None
    """Top-20 (location, count) pairs, or None if location absent."""

    remote_allowed_distribution: dict[str, int] | None
    """Value counts for remote_allowed, or None if absent."""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict representation."""
        d = asdict(self)
        # asdict converts list[tuple] correctly but be explicit for clarity
        d["top_20_skills"] = [
            {"skill": s, "count": c} for s, c in self.top_20_skills
        ]
        if self.location_top_20 is not None:
            d["location_top_20"] = [
                {"location": loc, "count": cnt}
                for loc, cnt in self.location_top_20
            ]
        return d


# ---------------------------------------------------------------------------
# Constants: candidate optional columns
# ---------------------------------------------------------------------------

_DATETIME_CANDIDATES: tuple[str, ...] = (
    "first_seen_at",
    "last_seen_at",
    "listed_time",
    "expiry",
    "closed_time",
    "original_listed_time",
    "application_deadline",
)

_SALARY_CANDIDATES: tuple[str, ...] = (
    "salary_min",
    "salary_max",
    "min_salary",
    "max_salary",
    "med_salary",
    "pay_period",
    "compensation_type",
    "currency",
    "normalized_salary",
)

_LOCATION_CANDIDATES: tuple[str, ...] = (
    "location",
    "state",
    "city",
    "zip_code",
    "country",
    "formatted_work_type",
    "work_type",
    "remote_allowed",
    "formatted_experience_level",
    "experience_level",
)


# ---------------------------------------------------------------------------
# Core profiling function
# ---------------------------------------------------------------------------


def profile_silver(df: pd.DataFrame) -> SilverProfile:
    """
    Compute a data profile for a silver Parquet DataFrame.

    Parameters
    ----------
    df:
        DataFrame loaded from a silver ``jobs.parquet`` file.  Must contain
        at least ``job_id``, ``title``, ``skills_text``, and either
        ``description_text`` or legacy ``description``. Additional columns are examined if present.
        ``skills_text``.  Additional columns are examined if present.

    Returns
    -------
    SilverProfile
        Structured summary of the dataset.

    Raises
    ------
    ValueError
        If any required silver column is missing.
    """
    _validate_required_columns(df)

    n_postings = len(df)
    n_unique_job_ids = int(df["job_id"].nunique())

    # --- missing counts ---
    n_missing_titles = int(
        df["title"].fillna("").astype(str).str.strip().eq("").sum()
    )
    description_col = "description_text" if "description_text" in df.columns else "description"
    n_missing_descriptions = int(
        df[description_col].fillna("").astype(str).str.strip().eq("").sum()
    )
    n_jobs_without_skills = int(
        df["skills_text"].fillna("").astype(str).str.strip().eq("").sum()
    )

    # --- skill frequency ---
    n_unique_skills, top_20_skills = _compute_skill_stats(df["skills_text"])

    # --- datetime columns ---
    datetime_columns_present = [
        c for c in _DATETIME_CANDIDATES if c in df.columns
    ]

    # --- timestamp parse rates ---
    lt_parse_rate, lt_total, lt_parsed = _compute_ts_parse_rate(
        df, "listed_time"
    )
    exp_parse_rate, exp_total, exp_parsed = _compute_ts_parse_rate(
        df, "expiry"
    )
    ct_parse_rate, ct_total, ct_parsed = _compute_ts_parse_rate(
        df, "closed_time"
    )

    # --- salary columns ---
    salary_columns_present = [
        c for c in _SALARY_CANDIDATES if c in df.columns
    ]
    salary_non_null_counts: dict[str, int] = {
        c: int(df[c].notna().sum()) for c in salary_columns_present
    }

    # --- location / work-type / experience columns ---
    location_columns_present = [
        c for c in _LOCATION_CANDIDATES if c in df.columns
    ]

    # --- categorical distributions ---
    work_type_distribution = _value_counts_dict(df, "formatted_work_type")
    experience_level_distribution = _value_counts_dict(
        df, "formatted_experience_level"
    )
    location_top_20 = _top_n_tuples(df, "location", n=20)
    remote_allowed_distribution = _value_counts_dict(df, "remote_allowed")

    return SilverProfile(
        n_postings=n_postings,
        n_unique_job_ids=n_unique_job_ids,
        n_missing_titles=n_missing_titles,
        n_missing_descriptions=n_missing_descriptions,
        n_jobs_without_skills=n_jobs_without_skills,
        n_unique_skills=n_unique_skills,
        top_20_skills=top_20_skills,
        datetime_columns_present=datetime_columns_present,
        listed_time_parse_rate=lt_parse_rate,
        listed_time_n_total=lt_total,
        listed_time_n_parsed=lt_parsed,
        expiry_parse_rate=exp_parse_rate,
        expiry_n_total=exp_total,
        expiry_n_parsed=exp_parsed,
        closed_time_parse_rate=ct_parse_rate,
        closed_time_n_total=ct_total,
        closed_time_n_parsed=ct_parsed,
        salary_columns_present=salary_columns_present,
        salary_non_null_counts=salary_non_null_counts,
        location_columns_present=location_columns_present,
        work_type_distribution=work_type_distribution,
        experience_level_distribution=experience_level_distribution,
        location_top_20=location_top_20,
        remote_allowed_distribution=remote_allowed_distribution,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SILVER_REQUIRED = ("job_id", "title", "skills_text")


def _validate_required_columns(df: pd.DataFrame) -> None:
    present = set(df.columns)
    missing = sorted(set(_SILVER_REQUIRED) - present)
    if "description_text" not in present and "description" not in present:
        missing.append("description_text")
    if missing:
        raise ValueError(
            f"DataFrame is missing required silver columns: {missing}. "
            f"Columns present: {sorted(df.columns.tolist())}"
        )


def _compute_skill_stats(
    skills_series: pd.Series,
) -> tuple[int, list[tuple[str, int]]]:
    """
    Count individual skill tokens across the ``skills_text`` column.

    Tokens are split on commas and stripped of whitespace.  Empty strings
    and NaN are ignored.

    Returns
    -------
    (n_unique_skills, top_20)
        ``n_unique_skills`` – number of distinct non-empty skill tokens.
        ``top_20``          – list of (skill_name, count) sorted descending.
    """
    from collections import Counter

    counter: Counter[str] = Counter()
    for cell in skills_series.fillna("").astype(str):
        for token in cell.split(","):
            token = token.strip()
            if token:
                counter[token] += 1

    n_unique = len(counter)
    top_20 = [(skill, count) for skill, count in counter.most_common(20)]
    return n_unique, top_20


def _try_parse_ms_timestamp(value: Any) -> bool:
    """
    Return True if *value* can be interpreted as a valid Unix timestamp in
    milliseconds (13-digit integer or numeric string).

    Accepts integers, floats, and numeric strings.  Rejects non-numeric
    strings and NaN.
    """
    try:
        ts_ms = float(value)
        if ts_ms != ts_ms:  # NaN check
            return False
        ts_s = ts_ms / 1000.0
        # Sanity: year 2000 → 2040 range
        dt = datetime.fromtimestamp(ts_s, tz=timezone.utc)
        return 2000 <= dt.year <= 2040
    except (TypeError, ValueError, OSError, OverflowError):
        return False


def _compute_listed_time_parse_rate(
    df: pd.DataFrame,
) -> tuple[float | None, int | None, int | None]:
    """
    Compute the parse-success rate for the ``listed_time`` column.

    Returns
    -------
    (parse_rate, n_total, n_parsed)
        All three are None if ``listed_time`` is absent from *df*.
        *parse_rate* is in [0.0, 1.0]; 1.0 when n_total == 0.
    """
    return _compute_ts_parse_rate(df, "listed_time")


def _compute_ts_parse_rate(
    df: pd.DataFrame,
    col: str,
) -> tuple[float | None, int | None, int | None]:
    """
    Compute the parse-success rate for a named timestamp column.

    Parameters
    ----------
    df:
        DataFrame to inspect.
    col:
        Name of the column to examine.

    Returns
    -------
    (parse_rate, n_total, n_parsed)
        All three are None if *col* is absent from *df*.
    """
    if col not in df.columns:
        return None, None, None

    series = df[col].dropna()
    n_total = len(series)
    if n_total == 0:
        return 1.0, 0, 0

    n_parsed = int(sum(_try_parse_ms_timestamp(v) for v in series))
    parse_rate = round(n_parsed / n_total, 6)
    return parse_rate, n_total, n_parsed


def _value_counts_dict(
    df: pd.DataFrame,
    col: str,
) -> dict[str, int] | None:
    """
    Return a value-counts dict for *col*, or None if *col* is absent.

    Non-null, non-empty values are counted.  Result is sorted descending by
    count.
    """
    if col not in df.columns:
        return None
    counts = (
        df[col]
        .fillna("")
        .astype(str)
        .str.strip()
        .pipe(lambda s: s[s != ""])
        .value_counts()
    )
    return {k: int(v) for k, v in counts.items()}


def _top_n_tuples(
    df: pd.DataFrame,
    col: str,
    n: int = 20,
) -> list[tuple[str, int]] | None:
    """
    Return top-*n* (value, count) tuples for *col*, or None if absent.
    """
    if col not in df.columns:
        return None
    counts = (
        df[col]
        .fillna("")
        .astype(str)
        .str.strip()
        .pipe(lambda s: s[s != ""])
        .value_counts()
        .head(n)
    )
    return [(k, int(v)) for k, v in counts.items()]


# ---------------------------------------------------------------------------
# Convenience: profile from file path
# ---------------------------------------------------------------------------


def profile_silver_from_path(silver_path: Path | str) -> SilverProfile:
    """
    Load a silver Parquet file and return its :class:`SilverProfile`.

    Parameters
    ----------
    silver_path:
        Path to a ``jobs.parquet`` silver file.

    Returns
    -------
    SilverProfile
    """
    path = Path(silver_path)
    if not path.exists():
        raise FileNotFoundError(f"Silver Parquet not found: {path}")
    df = pd.read_parquet(path)
    logger.info("Loaded silver Parquet for profiling: %s (%d rows)", path, len(df))
    return profile_silver(df)
