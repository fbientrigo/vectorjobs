"""Tests for agent labeling pack module (synthetic data only)."""

import json
from pathlib import Path
import pandas as pd
import pytest

from jobsrec.extract.agent_pack import (
    build_agent_labeling_pack,
    JSON_SCHEMA,
    PROMPT_MARKDOWN,
)


def _make_candidates(n: int = 100) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({
            "job_id": str(i % 10),
            "candidate_index": i // 10,
            "candidate_text": f"text candidate with some info {i}",
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
        "job_card_text": [f"Job Card Text for Job {i}" for i in range(n_jobs)],
        "description_text": [f"Description Text for Job {i}" for i in range(n_jobs)],
    })


def test_build_agent_labeling_pack_success():
    silver = _make_silver()
    candidates = _make_candidates()

    rows, manifest = build_agent_labeling_pack(
        silver=silver,
        candidates=candidates,
        sample_size=20,
        random_seed=42,
    )

    assert len(rows) == 20
    assert manifest["output_rows_count"] == 20
    assert manifest["random_seed"] == 42
    assert manifest["sample_size"] == 20

    # Verify keys in JSONL rows
    required_keys = {
        "task_id", "job_id", "candidate_index", "candidate_text",
        "candidate_source", "section_name", "title_clean", "company_name",
        "company_industry", "skills_regex_raw", "skills_normalized",
        "context", "baseline", "label_options", "expected_response_schema"
    }
    for row in rows:
        assert set(row.keys()) == required_keys
        assert isinstance(row["task_id"], str)
        assert isinstance(row["job_id"], str)
        assert isinstance(row["candidate_index"], int)
        assert isinstance(row["skills_regex_raw"], list)
        assert isinstance(row["skills_normalized"], list)
        assert isinstance(row["context"], dict)
        assert "job_card_text" in row["context"]
        assert "description_text_excerpt" in row["context"]
        assert isinstance(row["baseline"], dict)
        assert "regex_has_skill" in row["baseline"]
        assert isinstance(row["baseline"]["regex_has_skill"], bool)
        assert isinstance(row["baseline"]["regex_skills"], list)


def test_deterministic_sampling():
    silver = _make_silver()
    candidates = _make_candidates(150)

    rows1, _ = build_agent_labeling_pack(silver, candidates, sample_size=50, random_seed=123)
    rows2, _ = build_agent_labeling_pack(silver, candidates, sample_size=50, random_seed=123)
    rows3, _ = build_agent_labeling_pack(silver, candidates, sample_size=50, random_seed=456)

    # Identical seed -> identical results
    assert [r["task_id"] for r in rows1] == [r["task_id"] for r in rows2]
    # Different seed -> different results
    assert [r["task_id"] for r in rows1] != [r["task_id"] for r in rows3]


def test_json_schema_valid():
    assert "$schema" in JSON_SCHEMA
    assert JSON_SCHEMA["title"] == "AgentLabelingRow"


def test_prompt_markdown_valid():
    assert "HARD_SKILL" in PROMPT_MARKDOWN
    assert "UNCERTAIN" in PROMPT_MARKDOWN


def test_cli_integration(tmp_path):
    from click.testing import CliRunner
    from jobsrec.cli import main

    silver = _make_silver()
    candidates = _make_candidates()

    silver_path = tmp_path / "silver.parquet"
    candidates_path = tmp_path / "candidates.parquet"
    output_dir = tmp_path / "agent_labeling"

    silver.to_parquet(silver_path)
    candidates.to_parquet(candidates_path)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "build-agent-labeling-pack",
            "--silver-path", str(silver_path),
            "--candidates-path", str(candidates_path),
            "--output-dir", str(output_dir),
            "--sample-size", "25",
            "--random-seed", "3407",
        ],
    )

    assert result.exit_code == 0, f"CLI execution failed: {result.output}"

    # Verify expected outputs are created
    assert (output_dir / "agent_labeling_pack.jsonl").exists()
    assert (output_dir / "agent_labeling_manifest.json").exists()
    assert (output_dir / "agent_labeling_prompt.md").exists()
    assert (output_dir / "agent_labeling_schema.json").exists()

    # Load and check JSONL rows
    jsonl_lines = (output_dir / "agent_labeling_pack.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(jsonl_lines) == 25
    first_row = json.loads(jsonl_lines[0])
    assert "task_id" in first_row
    assert isinstance(first_row["skills_regex_raw"], list)
