"""Tests for M1 labeling seed (synthetic data only)."""
import json

import pandas as pd
import pytest

from jobsrec.extract.labeling import (
    LABEL_CLASSES,
    OUTPUT_COLUMNS,
    build_labeling_seed,
    write_labeling_seed,
)


def _make_candidates(n: int = 100) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({
            "job_id": str(i % 10),
            "candidate_index": i // 10,
            "candidate_text": f"text {i}",
            "candidate_source": ["title", "li", "paragraph"][i % 3],
            "section_name": ["", "requisitos", "habilidades"][i % 3],
            "skills_regex_raw": json.dumps(["python"] if i % 2 == 0 else []),
            "skills_normalized": json.dumps(["python"] if i % 2 == 0 else []),
        })
    return pd.DataFrame(rows)


def _make_silver(n_jobs: int = 10) -> pd.DataFrame:
    return pd.DataFrame({
        "job_id": [str(i) for i in range(n_jobs)],
        "title": [f"Job {i}" for i in range(n_jobs)],
        "company_name": [f"Company {i}" for i in range(n_jobs)],
        "company_industry": ["Salud" if i % 2 == 0 else "Retail" for i in range(n_jobs)],
    })


def test_label_constants_exist():
    assert "HARD_SKILL" in LABEL_CLASSES
    assert "IGNORE" in LABEL_CLASSES
    assert len(LABEL_CLASSES) == 12


def test_output_columns_exact():
    silver = _make_silver()
    candidates = _make_candidates()
    df = build_labeling_seed(silver, candidates, sample_size=20, random_seed=42)
    assert list(df.columns) == list(OUTPUT_COLUMNS)


def test_label_and_notes_empty():
    silver = _make_silver()
    candidates = _make_candidates()
    df = build_labeling_seed(silver, candidates, sample_size=20, random_seed=42)
    assert (df["label"] == "").all()
    assert (df["notes"] == "").all()


def test_sample_size_honored():
    silver = _make_silver()
    candidates = _make_candidates(200)
    df = build_labeling_seed(silver, candidates, sample_size=50, random_seed=42)
    assert len(df) == 50


def test_deduplication():
    silver = _make_silver()
    candidates = _make_candidates(100)
    df = build_labeling_seed(silver, candidates, sample_size=30, random_seed=42)
    dupes = df.duplicated(subset=["job_id", "candidate_index"])
    assert not dupes.any()


def test_deterministic():
    silver = _make_silver()
    candidates = _make_candidates(200)
    df1 = build_labeling_seed(silver, candidates, sample_size=50, random_seed=42)
    df2 = build_labeling_seed(silver, candidates, sample_size=50, random_seed=42)
    assert df1["job_id"].tolist() == df2["job_id"].tolist()
    assert df1["candidate_index"].tolist() == df2["candidate_index"].tolist()


def test_different_seeds_differ():
    silver = _make_silver()
    candidates = _make_candidates(200)
    df1 = build_labeling_seed(silver, candidates, sample_size=50, random_seed=42)
    df2 = build_labeling_seed(silver, candidates, sample_size=50, random_seed=99)
    assert df1["job_id"].tolist() != df2["job_id"].tolist()


def test_csv_writing(tmp_path):
    silver = _make_silver()
    candidates = _make_candidates(100)
    df = build_labeling_seed(silver, candidates, sample_size=20, random_seed=42)
    out = tmp_path / "seed.csv"
    write_labeling_seed(df, out)
    assert out.exists()
    loaded = pd.read_csv(out)
    assert list(loaded.columns) == list(OUTPUT_COLUMNS)
    assert len(loaded) == 20
