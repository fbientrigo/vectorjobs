"""Synthetic-data tests for the JobMarketGold v0 producer."""

from __future__ import annotations

import json

import pandas as pd

from jobsrec.ingest.gold_contract import (
    RETRIEVAL_CORPUS_COLUMNS,
    SKILL_SHARE_COLUMNS,
    build_gold_contract,
)


def _synthetic_silver() -> pd.DataFrame:
    # 2024-01 in Unix-ms; one row left without a time / url to exercise fallbacks.
    # Titles/descriptions are Spanish: role_family keywords match real Chilean
    # postings (see docs/specs/01_data_contract.md), not English LinkedIn text.
    jan_2024_ms = 1704067200000
    return pd.DataFrame(
        [
            {
                "job_id": 1,
                "title": "Ingeniero Civil de Obras",
                "description": "Diseno de puentes usando AutoCAD.",
                "skills_text": "",  # bronze has no skills table -> regex fallback
                "job_card_text": "Title: Ingeniero Civil de Obras\nDescription: Diseno de puentes.",
                "application_url": "https://example.com/1",
                "location": "",
                "company_region": "Region Metropolitana",
                "formatted_work_type": "Full-time",
                "formatted_experience_level": "Mid-Senior level",
                "listed_time": jan_2024_ms,
            },
            {
                "job_id": 2,
                "title": "Desarrollador de Software",
                "description": "Se requiere manejo de Python y SQL.",
                "skills_text": "",
                "job_card_text": "Title: Desarrollador de Software\nSkills: Python, SQL",
                "application_url": "https://example.com/2",
                "location": "",
                "company_region": "Region Metropolitana",
                "formatted_work_type": "Full-time",
                "formatted_experience_level": "Associate",
                "listed_time": jan_2024_ms,
            },
            {
                "job_id": 3,
                "title": "Ingeniero Industrial de Procesos",
                "description": "Gestion de operaciones en planta.",
                "skills_text": "",  # blank skills -> no skill rows
                "job_card_text": "",  # blank -> exercise builder fallback
                "application_url": None,  # missing url -> null
                "location": "",
                "company_region": "Region de Valparaiso",
                "formatted_work_type": "Full-time",
                "formatted_experience_level": "Director",
                "listed_time": None,  # missing time -> period "unknown"
            },
        ]
    )


def test_build_gold_contract_writes_full_contract(tmp_path):
    silver = tmp_path / "silver.parquet"
    _synthetic_silver().to_parquet(silver, index=False)
    out = tmp_path / "out"

    manifest = build_gold_contract(silver, out, max_rows=10)

    # 1. all four files exist
    for name in (
        "retrieval_corpus.parquet",
        "skill_share_by_period.parquet",
        "role_families.json",
        "dataset_manifest.json",
    ):
        assert (out / name).exists(), name

    # 2. retrieval corpus
    corpus = pd.read_parquet(out / "retrieval_corpus.parquet")
    assert len(corpus) >= 1
    assert list(corpus.columns) == RETRIEVAL_CORPUS_COLUMNS
    # fallback builder produced text for the blank-job_card_text row
    assert corpus["text"].str.len().gt(0).all()
    # missing url is null, not invented
    assert corpus.loc[corpus["doc_id"] == "3", "url"].isna().all()

    # 3. skill share -- skills_text is blank, so this exercises the regex
    # fallback over description_text (jobsrec.extract.skills.match_skills).
    skills = pd.read_parquet(out / "skill_share_by_period.parquet")
    assert list(skills.columns) == SKILL_SHARE_COLUMNS
    assert (skills["confidence"] == "low").all()
    assert "AutoCAD" in set(skills["skill_name"])
    assert "SQL" in set(skills["skill_name"])

    # 4. role families
    role_families = json.loads((out / "role_families.json").read_text(encoding="utf-8"))
    assert len(role_families["role_families"]) >= 1
    by_id = {f["role_family_id"]: f for f in role_families["role_families"]}
    assert by_id["civil_infrastructure"]["posting_count"] == 1
    assert by_id["computing_software"]["posting_count"] == 1
    assert by_id["industrial_management"]["posting_count"] == 1
    # location falls back to company_region since `location` is blank, like the
    # real silver dataset (location is nullable there -- see data contract).
    assert by_id["civil_infrastructure"]["locations"] == ["Region Metropolitana"]
    assert by_id["industrial_management"]["locations"] == ["Region de Valparaiso"]
    assert by_id["civil_infrastructure"]["sample_of_total_postings"] == 3

    # 5. manifest references all artifacts
    assert set(manifest["artifacts"]) == {
        "retrieval_corpus.parquet",
        "skill_share_by_period.parquet",
        "role_families.json",
        "dataset_manifest.json",
    }

    # 6. temporal confidence is low
    assert manifest["temporal_confidence"] == "low"
