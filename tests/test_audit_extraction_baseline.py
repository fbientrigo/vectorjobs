"""Tests for M0.7 baseline extraction quality audit."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from jobsrec.extract.audit import (
    AUDIT_VERSION,
    build_stratified_sample,
    compute_audit_metrics,
    write_json_report,
    write_markdown_report,
)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

_SILVER_ROWS = [
    {"job_id": "j-001", "title": "Analista SQL", "company_industry": "Tech", "company_parse_error": False},
    {"job_id": "j-002", "title": "Vendedor Retail", "company_industry": "Retail", "company_parse_error": False},
    {"job_id": "j-003", "title": "Sin skills", "company_industry": "Salud", "company_parse_error": True},
]

_CAND_ROWS = [
    # j-001 has SQL skill
    {
        "job_id": "j-001", "candidate_index": 0,
        "candidate_text": "Analista SQL",
        "candidate_source": "title", "section_name": "",
        "skills_regex_raw": json.dumps(["SQL"]),
        "skills_normalized": json.dumps(["SQL"]),
    },
    {
        "job_id": "j-001", "candidate_index": 1,
        "candidate_text": "Manejo de Excel avanzado.",
        "candidate_source": "li", "section_name": "requisitos",
        "skills_regex_raw": json.dumps(["Excel"]),
        "skills_normalized": json.dumps(["Excel"]),
    },
    # j-002 has no skills
    {
        "job_id": "j-002", "candidate_index": 0,
        "candidate_text": "Vendedor Retail",
        "candidate_source": "title", "section_name": "",
        "skills_regex_raw": json.dumps([]),
        "skills_normalized": json.dumps([]),
    },
    # j-003 has no skills
    {
        "job_id": "j-003", "candidate_index": 0,
        "candidate_text": "Sin descripción.",
        "candidate_source": "paragraph", "section_name": "",
        "skills_regex_raw": json.dumps([]),
        "skills_normalized": json.dumps([]),
    },
]


@pytest.fixture(scope="module")
def silver() -> pd.DataFrame:
    return pd.DataFrame(_SILVER_ROWS)


@pytest.fixture(scope="module")
def candidates() -> pd.DataFrame:
    return pd.DataFrame(_CAND_ROWS)


@pytest.fixture(scope="module")
def metrics(silver: pd.DataFrame, candidates: pd.DataFrame) -> dict:
    return compute_audit_metrics(silver, candidates)


# ---------------------------------------------------------------------------
# Metrics correctness
# ---------------------------------------------------------------------------

def test_total_jobs(metrics: dict) -> None:
    assert metrics["total_jobs"] == 3


def test_total_candidates(metrics: dict) -> None:
    assert metrics["total_candidates"] == 4


def test_jobs_with_candidates(metrics: dict) -> None:
    assert metrics["jobs_with_candidates"] == 3


def test_jobs_with_regex_skills(metrics: dict) -> None:
    assert metrics["jobs_with_regex_skills"] == 1  # only j-001


def test_jobs_with_regex_skills_pct(metrics: dict) -> None:
    assert metrics["jobs_with_regex_skills_pct"] == pytest.approx(33.3, abs=0.1)


def test_skill_counts_json_parsed(metrics: dict) -> None:
    # skills_normalized parsed via json.loads, not comma split
    assert metrics["skill_counts"]["SQL"] == 1
    assert metrics["skill_counts"]["Excel"] == 1


def test_candidate_source_counts(metrics: dict) -> None:
    assert metrics["candidate_counts_by_source"]["title"] == 2
    assert metrics["candidate_counts_by_source"]["li"] == 1
    assert metrics["candidate_counts_by_source"]["paragraph"] == 1


def test_section_skill_counts(metrics: dict) -> None:
    # Excel is under "requisitos", SQL has section_name=""
    assert metrics["skill_counts_by_section"]["requisitos"] == 1
    assert metrics["skill_counts_by_section"][""] == 1


def test_company_parse_errors(metrics: dict) -> None:
    assert metrics["company_parse_errors"] == 1


def test_industry_coverage_present(metrics: dict) -> None:
    ind = metrics["jobs_with_skills_by_industry"]
    assert "Tech" in ind
    assert ind["Tech"]["total_jobs"] == 1
    assert ind["Tech"]["jobs_with_skills"] == 1
    assert ind["Tech"]["pct"] == 100.0
    assert ind["Retail"]["jobs_with_skills"] == 0


def test_top_50_texts_lowercased(metrics: dict) -> None:
    texts = metrics["top_50_candidate_texts"]
    # All keys should be lowercase
    for key in texts:
        assert key == key.lower()


def test_mean_median_positive(metrics: dict) -> None:
    assert metrics["mean_candidates_per_job"] > 0
    assert metrics["median_candidates_per_job"] > 0


def test_schema_versions_present(metrics: dict) -> None:
    assert metrics["silver_schema_version"]
    assert metrics["extraction_schema_version"]
    assert metrics["skill_dict_version"]


# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------

def test_write_json_report(tmp_path: Path, metrics: dict) -> None:
    out = tmp_path / "report.json"
    write_json_report(metrics, out)
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["total_jobs"] == 3
    assert loaded["audit_version"] == AUDIT_VERSION


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def test_write_markdown_report(tmp_path: Path, metrics: dict) -> None:
    out = tmp_path / "report.md"
    write_markdown_report(metrics, out)
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "# Baseline Extraction Quality Audit" in content
    assert "## Top Skills" in content
    assert "## Candidate Sources" in content
    assert "## Warnings" in content


# ---------------------------------------------------------------------------
# Stratified sample
# ---------------------------------------------------------------------------

def test_sample_respects_size(silver: pd.DataFrame, candidates: pd.DataFrame) -> None:
    sample = build_stratified_sample(silver, candidates, sample_size=3)
    assert len(sample) <= 3


def test_sample_fills_requested_size_when_available(
    silver: pd.DataFrame, candidates: pd.DataFrame
) -> None:
    bigger = pd.concat(
        [
            candidates.assign(job_id=f"j-extra-{idx}", candidate_index=idx)
            for idx in range(10)
        ],
        ignore_index=True,
    )
    sample = build_stratified_sample(silver, bigger, sample_size=8)
    assert len(sample) == 8


def test_sample_no_duplicate_rows(silver: pd.DataFrame, candidates: pd.DataFrame) -> None:
    sample = build_stratified_sample(silver, candidates, sample_size=100)
    assert not sample.duplicated(subset=["job_id", "candidate_index"]).any()


def test_sample_includes_has_skills(silver: pd.DataFrame, candidates: pd.DataFrame) -> None:
    sample = build_stratified_sample(silver, candidates, sample_size=100)
    has_skills = sample["skills_normalized"].apply(lambda x: bool(json.loads(x)))
    assert has_skills.any()


def test_sample_includes_title_source(silver: pd.DataFrame, candidates: pd.DataFrame) -> None:
    sample = build_stratified_sample(silver, candidates, sample_size=100)
    assert (sample["candidate_source"] == "title").any()


def test_sample_empty_candidates(silver: pd.DataFrame) -> None:
    empty = pd.DataFrame(columns=list(_CAND_ROWS[0].keys()))
    sample = build_stratified_sample(silver, empty, sample_size=10)
    assert len(sample) == 0
