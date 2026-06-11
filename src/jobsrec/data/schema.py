"""
Column-level schema contracts for every CSV / Parquet consumed by jobsrec.

Validation is intentionally lightweight (no Pydantic / pandera dependency)
so the package runs in constrained Colab environments without heavy extras.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import pandas as pd


# ---------------------------------------------------------------------------
# Column contracts
# ---------------------------------------------------------------------------

#: Required columns for postings.csv
POSTINGS_REQUIRED: tuple[str, ...] = ("job_id", "title", "description")

#: Optional columns carried through to the silver dataset.
#: All are preserved as-is when present in postings.csv.
#: Temporal columns (listed_time, original_listed_time, expiry, closed_time)
#: are kept in their raw numeric form (Unix-ms integer) without casting.
POSTINGS_OPTIONAL: tuple[str, ...] = (
    # --- temporal ---
    "listed_time",
    "original_listed_time",
    "expiry",
    "closed_time",
    # --- job metadata ---
    "formatted_work_type",
    "formatted_experience_level",
    "work_type",
    "location",
    "remote_allowed",
    "skills_desc",
    "sponsored",
    # --- compensation (from postings; may be superseded by salaries join) ---
    "normalized_salary",
    # --- application ---
    "application_url",
    "application_type",
    "views",
    "applies",
    # --- company ---
    "company_id",
    # --- geo ---
    "zip_code",
    "fips",
)

#: Required columns for jobs/job_skills.csv
JOB_SKILLS_REQUIRED: tuple[str, ...] = ("job_id", "skill_abr")

#: Required columns for mappings/skills.csv
SKILLS_REQUIRED: tuple[str, ...] = ("skill_abr", "skill_name")

#: Required columns for jobs/salaries.csv
SALARIES_REQUIRED: tuple[str, ...] = ("job_id",)

#: Salary columns brought in from jobs/salaries.csv (all optional).
#: If multiple rows per job_id exist they are aggregated (first non-null wins
#: for string fields; numeric fields take the max of non-null values).
SALARIES_OPTIONAL: tuple[str, ...] = (
    "min_salary",
    "max_salary",
    "med_salary",
    "pay_period",
    "currency",
    "compensation_type",
)

#: Required columns in the silver Parquet output
SILVER_REQUIRED: tuple[str, ...] = (
    "job_id",
    "title",
    "description",
    "skills_text",
    "job_card_text",
)


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ValidationResult:
    """Outcome of a schema validation check."""

    valid: bool
    missing_columns: list[str] = field(default_factory=list)
    extra_columns: list[str] = field(default_factory=list)
    source: str = ""

    def __str__(self) -> str:
        if self.valid:
            return f"ValidationResult(valid=True, source={self.source!r})"
        return (
            f"ValidationResult(valid=False, source={self.source!r}, "
            f"missing={self.missing_columns})"
        )


def validate_columns(
    df: pd.DataFrame,
    required: Sequence[str],
    source: str = "",
) -> ValidationResult:
    """
    Check that *df* contains every column listed in *required*.

    Parameters
    ----------
    df:
        DataFrame to inspect.
    required:
        Column names that must be present.
    source:
        Human-readable label used in error messages (e.g. ``"postings.csv"``).

    Returns
    -------
    ValidationResult
        Always returns a result object; callers decide whether to raise.
    """
    present = set(df.columns)
    required_set = set(required)
    missing = sorted(required_set - present)
    extra = sorted(present - required_set)
    return ValidationResult(
        valid=len(missing) == 0,
        missing_columns=missing,
        extra_columns=extra,
        source=source,
    )


def assert_columns(
    df: pd.DataFrame,
    required: Sequence[str],
    source: str = "",
) -> None:
    """
    Like :func:`validate_columns` but raises ``ValueError`` on failure.

    Parameters
    ----------
    df:
        DataFrame to inspect.
    required:
        Column names that must be present.
    source:
        Human-readable label used in the error message.

    Raises
    ------
    ValueError
        If any required column is missing.
    """
    result = validate_columns(df, required, source=source)
    if not result.valid:
        raise ValueError(
            f"[{source or 'DataFrame'}] Missing required columns: "
            f"{result.missing_columns}.  "
            f"Columns present: {sorted(df.columns.tolist())}"
        )
