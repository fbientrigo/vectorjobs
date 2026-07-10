import json

import pytest

import evaluate_against_gold as evg


def test_exits_cleanly_when_no_gold_file(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["evaluate_against_gold.py", "--gold", str(tmp_path / "missing.jsonl")])
    with pytest.raises(SystemExit) as exc_info:
        evg.main()
    assert exc_info.value.code == 0


def test_exits_cleanly_when_gold_is_all_pending(tmp_path, monkeypatch):
    gold_path = tmp_path / "gold.jsonl"
    gold_path.write_text(json.dumps({"job_id": "1", "review_status": "pending"}) + "\n", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["evaluate_against_gold.py", "--gold", str(gold_path)])
    with pytest.raises(SystemExit) as exc_info:
        evg.main()
    assert exc_info.value.code == 0


def test_evaluate_computes_exact_f1_on_synthetic_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(evg, "OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.setattr(evg, "SAMPLE_PATH", tmp_path / "sample.jsonl")

    (tmp_path / "sample.jsonl").write_text(
        json.dumps({"job_id": "1", "job_card_text": "Se requiere manejo de Python y Excel."}) + "\n",
        encoding="utf-8",
    )

    model_dir = tmp_path / "outputs" / "fake__model-a"
    model_dir.mkdir(parents=True)
    extraction_row = {
        "job_id": "1",
        "apolo_extraction": {
            "skills": [
                {"name": "Python", "normalized_name": "python", "category": "technical", "evidence": "manejo de Python"},
                {"name": "Excel", "normalized_name": "excel", "category": "tool", "evidence": "y Excel"},
            ],
            "education_requirements": [],
            "seniority": "junior",
        },
    }
    (model_dir / "extractions.jsonl").write_text(json.dumps(extraction_row) + "\n", encoding="utf-8")

    gold_rows = [
        {
            "job_id": "1",
            "review_status": "reviewed",
            # "python" matches the prediction exactly; "sql" is a gold skill the model missed.
            "gold_skills": [
                {"name": "python", "normalized_name": "python", "category": "technical"},
                {"name": "sql", "normalized_name": "sql", "category": "technical"},
            ],
            "gold_education_requirements": [],
            "seniority_gold": "junior",
        }
    ]

    result = evg.evaluate("fake/model-a", gold_rows)

    # tp=1 (python), fp=1 (excel), fn=1 (sql) -> precision=0.5, recall=0.5, f1=0.5
    assert result["skill_name_exact_precision"] == pytest.approx(0.5)
    assert result["skill_name_exact_recall"] == pytest.approx(0.5)
    assert result["skill_name_exact_f1"] == pytest.approx(0.5)
    assert result["seniority_accuracy"] == pytest.approx(1.0)
    assert result["evidence_substring_failure_count"] == 0
    assert result["n_jobs_evaluated"] == 1
