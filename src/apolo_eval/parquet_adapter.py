"""Parquet adapter for producing deterministic job-evaluation texts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class JobTextRecord:
    text: str
    id: str | None = None


def load_job_texts_from_parquet(
    path: str | Path,
    title_column: str,
    description_column: str,
    skills_column: str | None = None,
    id_column: str | None = None,
    limit: int | None = None,
) -> list[JobTextRecord]:
    df = pd.read_parquet(path)
    _require_columns(df.columns, [title_column, description_column])
    optional = [col for col in [skills_column, id_column] if col is not None]
    _require_columns(df.columns, optional)

    if limit is not None:
        df = df.head(limit)

    records: list[JobTextRecord] = []
    for _, row in df.iterrows():
        text = _format_job_text(
            title=_clean(row[title_column]),
            description=_clean(row[description_column]),
            skills=_clean(row[skills_column]) if skills_column is not None else "",
        )
        record_id = _clean(row[id_column]) if id_column is not None else None
        records.append(JobTextRecord(text=text, id=record_id or None))
    return records


def _format_job_text(title: str, description: str, skills: str) -> str:
    lines = [f"TITLE: {title}"]
    if skills:
        lines.append(f"SKILLS: {skills}")
    lines.append(f"DESCRIPTION: {description}")
    return "\n".join(lines)


def _clean(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).split())


def _require_columns(columns: Any, required: list[str]) -> None:
    missing = [column for column in required if column not in columns]
    if missing:
        raise ValueError(f"missing required parquet column(s): {', '.join(missing)}")
