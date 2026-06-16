from pathlib import Path

import pandas as pd
import pytest

from apolo_eval.parquet_adapter import load_job_texts_from_parquet


def test_parquet_adapter_loads_deterministic_texts(tmp_path: Path) -> None:
    path = tmp_path / "jobs.parquet"
    pd.DataFrame(
        {
            "job_id": [101, 102],
            "title": ["Data Engineer", "Frontend Developer"],
            "description": [
                "Build reliable ETL pipelines.",
                "Create React interfaces.",
            ],
            "skills": ["Python, SQL", "React, TypeScript"],
        }
    ).to_parquet(path)

    records = load_job_texts_from_parquet(
        path,
        title_column="title",
        description_column="description",
        skills_column="skills",
        id_column="job_id",
    )

    assert [record.id for record in records] == ["101", "102"]
    assert records[0].text == (
        "TITLE: Data Engineer\n"
        "SKILLS: Python, SQL\n"
        "DESCRIPTION: Build reliable ETL pipelines."
    )


def test_parquet_adapter_allows_missing_skills_column_when_not_configured(
    tmp_path: Path,
) -> None:
    path = tmp_path / "jobs.parquet"
    pd.DataFrame(
        {
            "title": ["Analyst"],
            "description": ["Build dashboards."],
        }
    ).to_parquet(path)

    records = load_job_texts_from_parquet(
        path,
        title_column="title",
        description_column="description",
    )

    assert records[0].text == "TITLE: Analyst\nDESCRIPTION: Build dashboards."


def test_parquet_adapter_reports_missing_required_columns(tmp_path: Path) -> None:
    path = tmp_path / "jobs.parquet"
    pd.DataFrame({"title": ["Analyst"]}).to_parquet(path)

    with pytest.raises(ValueError, match="description"):
        load_job_texts_from_parquet(
            path,
            title_column="title",
            description_column="description",
        )
