from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from jobsrec.data.profile import (
    _compute_listed_time_parse_rate,
    profile_silver,
    profile_silver_from_path,
)


def make_silver_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "job_id": ["a", "b", "c"],
            "title": ["A", "B", ""],
            "description_text": ["Build pipelines.", "", "Analyse data."],
            "skills_text": ["Python, SQL", "", ""],
            "job_card_text": ["x", "y", "z"],
            "first_seen_at": ["2026-03-31", "2026-04-01", None],
        }
    )


def test_profile_accepts_description_text() -> None:
    profile = profile_silver(make_silver_df())
    assert profile.n_postings == 3
    assert profile.n_unique_job_ids == 3
    assert profile.n_missing_titles == 1
    assert profile.n_missing_descriptions == 1
    assert profile.n_jobs_without_skills == 2
    assert profile.n_unique_skills == 2


def test_profile_from_path(tmp_path: Path) -> None:
    path = tmp_path / "jobs.parquet"
    make_silver_df().to_parquet(path, index=False)
    assert profile_silver_from_path(path).n_postings == 3


def test_profile_requires_description_text_or_legacy_description() -> None:
    df = make_silver_df().drop(columns=["description_text"])
    with pytest.raises(ValueError, match="description_text"):
        profile_silver(df)


def test_legacy_listed_time_parse_rate_still_works() -> None:
    df = make_silver_df()
    df["listed_time"] = [1_700_000_000_000, "not-a-date", None]
    rate, total, parsed = _compute_listed_time_parse_rate(df)
    assert rate == pytest.approx(0.5)
    assert total == 2
    assert parsed == 1
