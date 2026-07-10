import json
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parent.parent
sample_path = EXPERIMENT_DIR / "input" / "jobs_sample_50.jsonl"
out_path = EXPERIMENT_DIR / "gold" / "human_gold_template.jsonl"

rows = [json.loads(l) for l in sample_path.read_text(encoding="utf-8").splitlines() if l.strip()]
assert len(rows) == 50, len(rows)

with out_path.open("w", encoding="utf-8") as f:
    for job in rows:
        gold_row = {
            "job_id": job["job_id"],
            "title": job["title"],
            "job_card_text": job["job_card_text"],
            "reviewer_id": "",
            "review_status": "pending",
            "gold_skills": [],
            "gold_education_requirements": [],
            "seniority_gold": "unknown",
            "ambiguous_cases": [],
            "notes": "",
        }
        f.write(json.dumps(gold_row, ensure_ascii=False) + "\n")

print("wrote", out_path, "rows:", len(rows))
