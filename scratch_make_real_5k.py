from pathlib import Path
import pandas as pd

root = Path("data")
out = Path("scratch/real_5k")
(out / "jobs").mkdir(parents=True, exist_ok=True)
(out / "mappings").mkdir(parents=True, exist_ok=True)

postings = pd.read_csv(root / "postings.csv", nrows=5000)
job_ids = set(postings["job_id"].astype(str))

postings.to_csv(out / "postings.csv", index=False)

job_skills = pd.read_csv(root / "jobs" / "job_skills.csv")
job_skills[job_skills["job_id"].astype(str).isin(job_ids)].to_csv(out / "jobs" / "job_skills.csv", index=False)

skills = pd.read_csv(root / "mappings" / "skills.csv")
skills.to_csv(out / "mappings" / "skills.csv", index=False)

print("Wrote", out)
print("postings", len(postings))
