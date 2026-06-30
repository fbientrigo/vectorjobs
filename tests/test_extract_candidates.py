"""Tests for the extraction candidates builder."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from jobsrec.extract.candidates import (
    CANDIDATE_COLUMNS,
    EXTRACTION_SCHEMA_VERSION,
    build_extraction_candidates,
    build_extraction_report,
    write_extraction_manifest,
)


SILVER_ROWS = [
    {
        "job_id": "j-001",
        "title": "Analista de Datos con Python y SQL",
        "description_html": (
            "<p>Responsabilidades principales:</p>"
            "<ul>"
            "  <li>Desarrollar dashboards en Power BI.</li>"
            "  <li>Consultar bases de datos SQL.</li>"
            "</ul>"
            "<p>Requisitos:</p>"
            "<ul>"
            "  <li>Manejo de Excel avanzado.</li>"
            "  <li>Inglés intermedio.</li>"
            "</ul>"
        ),
    },
    {
        "job_id": "j-002",
        "title": "Ingeniero Civil en Obras",
        "description_html": "<p>Sin descripción.</p>",
    },
    {
        "job_id": "j-003",
        "title": "",
        "description_html": "",
    },
]


@pytest.fixture(scope="module")
def silver_df() -> pd.DataFrame:
    return pd.DataFrame(SILVER_ROWS)


@pytest.fixture(scope="module")
def candidates_df(silver_df: pd.DataFrame) -> pd.DataFrame:
    return build_extraction_candidates(silver_df)


def test_output_columns(candidates_df: pd.DataFrame) -> None:
    assert list(candidates_df.columns) == list(CANDIDATE_COLUMNS)


def test_candidate_index_is_int32(candidates_df: pd.DataFrame) -> None:
    assert candidates_df["candidate_index"].dtype == "int32"


def test_more_candidates_than_jobs(candidates_df: pd.DataFrame, silver_df: pd.DataFrame) -> None:
    assert len(candidates_df) > len(silver_df)


def test_all_job_ids_present(candidates_df: pd.DataFrame) -> None:
    # j-003 has no title and no html, so it may produce 0 rows
    present = set(candidates_df["job_id"].unique())
    assert "j-001" in present
    assert "j-002" in present


def test_title_row_present(candidates_df: pd.DataFrame) -> None:
    title_rows = candidates_df[candidates_df["candidate_source"] == "title"]
    assert "j-001" in title_rows["job_id"].values
    assert "j-002" in title_rows["job_id"].values


def test_li_rows_present(candidates_df: pd.DataFrame) -> None:
    li_rows = candidates_df[candidates_df["candidate_source"] == "li"]
    assert len(li_rows) >= 4


def test_section_carry_forward(candidates_df: pd.DataFrame) -> None:
    j1 = candidates_df[candidates_df["job_id"] == "j-001"]
    li_rows = j1[j1["candidate_source"] == "li"]
    # First two li items follow the "responsabilidades" paragraph
    first_li = li_rows.iloc[0]
    assert first_li["section_name"] == "responsabilidades"
    # Third li item follows the "requisitos" paragraph
    third_li = li_rows.iloc[2]
    assert third_li["section_name"] == "requisitos"


def test_skills_regex_raw_is_json_list(candidates_df: pd.DataFrame) -> None:
    for _, row in candidates_df.iterrows():
        parsed = json.loads(row["skills_regex_raw"])
        assert isinstance(parsed, list)
        for item in parsed:
            assert isinstance(item, str)


def test_skills_normalized_is_json_list(candidates_df: pd.DataFrame) -> None:
    for _, row in candidates_df.iterrows():
        parsed = json.loads(row["skills_normalized"])
        assert isinstance(parsed, list)
        for item in parsed:
            assert isinstance(item, str)


def test_title_with_skills_detected(candidates_df: pd.DataFrame) -> None:
    title_row = candidates_df[
        (candidates_df["job_id"] == "j-001") & (candidates_df["candidate_source"] == "title")
    ].iloc[0]
    normed = json.loads(title_row["skills_normalized"])
    assert "Python" in normed
    assert "SQL" in normed


def test_li_with_excel_detected(candidates_df: pd.DataFrame) -> None:
    li_rows = candidates_df[
        (candidates_df["job_id"] == "j-001") & (candidates_df["candidate_source"] == "li")
    ]
    skills_in_li = []
    for _, row in li_rows.iterrows():
        skills_in_li.extend(json.loads(row["skills_normalized"]))
    assert "Excel" in skills_in_li
    assert "Power BI" in skills_in_li


def test_empty_description_produces_only_title(candidates_df: pd.DataFrame) -> None:
    j2 = candidates_df[candidates_df["job_id"] == "j-002"]
    # j-002 has title + one paragraph "Sin descripción." — no li items
    sources = set(j2["candidate_source"].tolist())
    assert "li" not in sources


def test_extraction_schema_version_format() -> None:
    assert EXTRACTION_SCHEMA_VERSION.startswith("extraction_v")


def test_build_report_counts(silver_df: pd.DataFrame, candidates_df: pd.DataFrame) -> None:
    report = build_extraction_report(candidates_df, silver_df, parse_error_count=1)
    assert report["jobs_processed"] == 3
    assert report["candidate_rows"] == len(candidates_df)
    assert 0 < report["jobs_with_candidates_pct"] <= 100
    assert report["company_parse_errors"] == 1
    assert report["skill_dict_version"].startswith("v")
    assert isinstance(report["top_skills"], list)


def test_write_extraction_manifest(tmp_path: Path, silver_df: pd.DataFrame, candidates_df: pd.DataFrame) -> None:
    report = build_extraction_report(candidates_df, silver_df, parse_error_count=0)
    manifest_path = write_extraction_manifest(
        output_dir=tmp_path,
        report=report,
        silver_path="data/silver/jobs.parquet",
        output_path="data/silver/job_extraction_candidates.parquet",
    )
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["extraction_schema_version"] == EXTRACTION_SCHEMA_VERSION
    assert manifest["skill_dict_version"] == report["skill_dict_version"]
    assert "candidate_columns" in manifest
    assert "top_skills" in manifest
    assert isinstance(manifest["top_skills"], list)
    assert "created_at" in manifest
