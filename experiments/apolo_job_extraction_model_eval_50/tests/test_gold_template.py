import json
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parent.parent
SAMPLE_PATH = EXPERIMENT_DIR / "input" / "jobs_sample_50.jsonl"
GOLD_TEMPLATE_PATH = EXPERIMENT_DIR / "gold" / "human_gold_template.jsonl"

REQUIRED_FIELDS = {
    "job_id", "title", "reviewer_id", "review_status",
    "gold_skills", "gold_education_requirements", "seniority_gold",
    "ambiguous_cases", "notes",
}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_gold_template_has_50_rows_matching_frozen_sample():
    sample_rows = read_jsonl(SAMPLE_PATH)
    gold_rows = read_jsonl(GOLD_TEMPLATE_PATH)

    assert len(gold_rows) == 50
    assert [r["job_id"] for r in gold_rows] == [r["job_id"] for r in sample_rows]


def test_gold_template_rows_are_blind_and_pending():
    for row in read_jsonl(GOLD_TEMPLATE_PATH):
        assert REQUIRED_FIELDS.issubset(row.keys())
        assert row["review_status"] == "pending"
        assert row["gold_skills"] == []
        assert row["gold_education_requirements"] == []
        assert "apolo_extraction" not in row
        assert "model" not in row
