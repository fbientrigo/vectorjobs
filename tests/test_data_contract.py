from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from jobsrec.data.load import build_silver
from jobsrec.data.load import ParsedCompany, parse_company
from jobsrec.data.schema import (
    BRONZE_JOBS_REQUIRED,
    BRONZE_OBSERVATIONS_REQUIRED,
    SILVER_REQUIRED,
    SILVER_SCHEMA_VERSION,
    assert_columns,
    validate_columns,
)


def make_bronze_db(path: Path) -> None:
    con = sqlite3.connect(path)
    con.executescript(
        """
        create table jobs (
            id text primary key,
            title text,
            company text,
            location text,
            description text,
            status text
        );
        create table job_observations (
            job_id text,
            crawl_id integer,
            seen_at timestamp,
            primary key (job_id, crawl_id)
        );
        create table crawl_runs (
            id integer primary key,
            started_at timestamp
        );
        """
    )
    con.executemany(
        "insert into crawl_runs(id, started_at) values (?, ?)",
        [(1, "2026-03-31 08:00:00"), (2, "2026-04-01 08:00:00")],
    )
    con.executemany(
        "insert into jobs(id, title, company, location, description, status) values (?, ?, ?, ?, ?, ?)",
        [
            (
                "a-1",
                "Data Engineer",
                "{'confidencial': False, 'denominacion': 'Acme Corp', 'ciudad': 'Santiago', 'provincia': 'Metropolitana', 'industria': 'Tecnología'}",
                None,
                "<p>Build <strong>pipelines</strong>.</p>",
                "active",
            ),
            (
                "b-2",
                "",
                "{'confidencial': True}",
                "Santiago",
                "<ul><li>Useful deleted row</li></ul>",
                "deleted",
            ),
            ("drop-me", "", "not a dict", None, "", "deleted"),
        ],
    )
    con.executemany(
        "insert into job_observations(job_id, crawl_id, seen_at) values (?, ?, ?)",
        [
            ("a-1", 1, "2026-03-31 09:00:00"),
            ("a-1", 2, "2026-04-01 09:00:00"),
            ("b-2", 2, "2026-04-01 10:00:00"),
        ],
    )
    con.commit()
    con.close()


def test_schema_validator_accepts_bronze_columns() -> None:
    jobs = pd.DataFrame(columns=list(BRONZE_JOBS_REQUIRED) + ["extra"])
    observations = pd.DataFrame(columns=list(BRONZE_OBSERVATIONS_REQUIRED))

    assert validate_columns(jobs, BRONZE_JOBS_REQUIRED, source="jobs").valid
    assert validate_columns(
        observations, BRONZE_OBSERVATIONS_REQUIRED, source="job_observations"
    ).valid


def test_schema_validator_reports_missing_columns() -> None:
    df = pd.DataFrame({"id": ["1"]})
    with pytest.raises(ValueError, match="title"):
        assert_columns(df, BRONZE_JOBS_REQUIRED, source="jobs")


def test_build_silver_from_sqlite(tmp_path: Path) -> None:
    db_path = tmp_path / "jobs.db"
    out_dir = tmp_path / "silver"
    make_bronze_db(db_path)

    result = build_silver(db_path, out_dir)
    silver = pd.read_parquet(result.output_path)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert list(silver.columns) == list(SILVER_REQUIRED)
    assert silver["job_id"].tolist() == ["a-1", "b-2"]
    # company_name now reads from denominacion
    assert silver.loc[0, "company_name"] == "Acme Corp"
    assert not bool(silver.loc[0, "company_confidential"])
    assert silver.loc[0, "company_city"] == "Santiago"
    assert silver.loc[0, "company_region"] == "Metropolitana"
    assert silver.loc[0, "company_industry"] == "Tecnología"
    assert not bool(silver.loc[0, "company_parse_error"])
    assert silver.loc[1, "company_name"] == "Confidencial"
    assert not bool(silver.loc[1, "company_parse_error"])
    assert silver.loc[0, "description_text"] == "Build pipelines."
    assert silver.loc[1, "status"] == "deleted"
    assert silver.loc[0, "first_seen_at"] == "2026-03-31 09:00:00"
    assert silver.loc[0, "last_seen_at"] == "2026-04-01 09:00:00"
    assert silver.loc[0, "times_seen"] == 2
    assert silver.loc[0, "crawl_count"] == 2
    assert silver["skills_text"].tolist() == ["", ""]
    assert "Description: Build pipelines." in silver.loc[0, "job_card_text"]
    assert manifest["source_tables"] == {
        "jobs": 3,
        "job_observations": 3,
        "crawl_runs": 2,
    }
    assert manifest["skills_source"] == "none"
    assert manifest["silver_schema_version"] == SILVER_SCHEMA_VERSION


def test_parse_company_denominacion() -> None:
    pc = parse_company("{'denominacion': 'Banco Chile', 'ciudad': 'Valparaíso', 'provincia': 'Valparaíso', 'industria': 'Banca', 'confidencial': False}")
    assert pc.name == "Banco Chile"
    assert pc.city == "Valparaíso"
    assert pc.region == "Valparaíso"
    assert pc.industry == "Banca"
    assert pc.confidential is False
    assert not pc.parse_error


def test_parse_company_parse_error() -> None:
    pc = parse_company("not a valid python literal {{{")
    assert pc.parse_error is True
    assert pc.name == "not a valid python literal {{{"
    assert pc.city == ""


def test_parse_company_confidential_fallback() -> None:
    pc = parse_company("{'confidencial': True}")
    assert pc.name == "Confidencial"
    assert pc.confidential is True
    assert not pc.parse_error


def test_parse_company_nombre_fallback() -> None:
    pc = parse_company("{'nombre': 'Legacy Name', 'confidencial': False}")
    assert pc.name == "Legacy Name"
    assert not pc.parse_error
