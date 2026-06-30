"""Column-level schema contracts for jobsrec datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import pandas as pd


BRONZE_REQUIRED_TABLES: tuple[str, ...] = (
    "jobs",
    "job_observations",
    "crawl_runs",
)

BRONZE_JOBS_REQUIRED: tuple[str, ...] = (
    "id",
    "title",
    "company",
    "location",
    "description",
    "status",
)

BRONZE_OBSERVATIONS_REQUIRED: tuple[str, ...] = (
    "job_id",
    "crawl_id",
    "seen_at",
)

BRONZE_CRAWL_RUNS_REQUIRED: tuple[str, ...] = ("id", "started_at")

SILVER_SCHEMA_VERSION = "0.2.0"

SILVER_REQUIRED: tuple[str, ...] = (
    "job_id",
    "title",
    "company_name",
    "company_confidential",
    "company_raw",
    "company_city",
    "company_region",
    "company_industry",
    "company_parse_error",
    "location",
    "description_html",
    "description_text",
    "status",
    "first_seen_at",
    "last_seen_at",
    "times_seen",
    "crawl_count",
    "skills_text",
    "job_card_text",
)


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
    """Check that *df* contains every column listed in *required*."""
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
    """Like :func:`validate_columns` but raises ``ValueError`` on failure."""
    result = validate_columns(df, required, source=source)
    if not result.valid:
        raise ValueError(
            f"[{source or 'DataFrame'}] Missing required columns: "
            f"{result.missing_columns}.  "
            f"Columns present: {sorted(df.columns.tolist())}"
        )
