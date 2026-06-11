# Milestone 1 — Real Data Smoke Test Report

**Date:** 2026-06-08  
**Environment:** Python 3.13.5, Windows 11, CPU-only  
**Fixture:** `tests/fixtures/kaggle_minimal/` (10 synthetic postings)

---

## Commands Run

### 1. Build Silver

```
python -m jobsrec.cli build-silver \
    --input-dir tests/fixtures/kaggle_minimal \
    --output-dir scratch/m1_silver
```

**Output:**
```json
{
  "output_path": "scratch\\m1_silver\\jobs.parquet",
  "input_rows": 10,
  "output_rows": 10
}
```

### 2. Build TF-IDF

```
python -m jobsrec.cli build-tfidf \
    --silver-path scratch/m1_silver/jobs.parquet \
    --output-dir scratch/m1_gold
```

**Output:**
```json
{
  "vectorizer_path": "scratch\\m1_gold\\tfidf_vectorizer.joblib",
  "matrix_path": "scratch\\m1_gold\\tfidf_matrix.npz",
  "n_docs": 10,
  "vocab_size": 55
}
```

### 3. Profile Silver

```
python -m jobsrec.cli profile-silver \
    --silver-path scratch/m1_silver/jobs.parquet \
    --output reports/m1_data_profile.json
```

**Output:**
```
Profile written to: reports\m1_data_profile.json
  n_postings          : 10
  n_unique_job_ids    : 10
  n_missing_titles    : 0
  n_jobs_without_skills: 0
  n_unique_skills     : 21
```

### 4. Recommend

```
python -m jobsrec.cli recommend \
    --job-id 1001 \
    --top-k 3 \
    --gold-dir scratch/m1_gold
```

**Output:**
```json
{
  "query_job_id": 1001,
  "results": [
    {"rank": 1, "job_id": 1004, "score": 0.773467},
    {"rank": 2, "job_id": 1006, "score": 0.55168},
    {"rank": 3, "job_id": 1008, "score": 0.512281}
  ]
}
```

---

## Test Results

```
============================= test session starts =============================
platform win32 -- Python 3.13.5, pytest-9.0.2
collected 86 items

tests/test_data_contract.py ...............              [ 17%]
tests/test_job_card.py .................                 [ 37%]
tests/test_m1_profile_and_smoke.py .................................  [ 75%]
tests/test_tfidf_retrieval.py .....................               [100%]

86 passed in 5.98s
```

**M0 (original) tests:** 53 passed  
**M1 new tests:** 33 passed  
**Total:** 86 passed, 0 failed

---

## Files Changed / Added

| File | Status | Notes |
|------|--------|-------|
| `src/jobsrec/data/profile.py` | **NEW** | Data profiling module |
| `src/jobsrec/cli.py` | **MODIFIED** | Added `profile-silver` command |
| `tests/test_m1_profile_and_smoke.py` | **NEW** | 33 M1 tests |
| `tests/fixtures/kaggle_minimal/postings.csv` | **NEW** | 10-posting synthetic fixture |
| `tests/fixtures/kaggle_minimal/jobs/job_skills.csv` | **NEW** | 26 skill assignments |
| `tests/fixtures/kaggle_minimal/mappings/skills.csv` | **NEW** | 21 skill definitions |
| `reports/m1_data_profile.json` | **NEW** | Generated data profile |
| `reports/m1_real_data_smoke_report.md` | **NEW** | This report |
| `README.md` | **MODIFIED** | Added M1 section + profile-silver docs |

---

## Profile Output Summary

From `reports/m1_data_profile.json`:

| Metric | Value |
|--------|-------|
| Total postings | 10 |
| Unique job IDs | 10 |
| Missing titles | 0 |
| Missing descriptions | 0 |
| Jobs without skills | 0 |
| Unique skills | 21 |
| Top skill (Python) | count: 3 |
| Top skill (Docker) | count: 3 |
| location_columns_present | location, formatted_work_type, formatted_experience_level |
| salary_columns_present | [] *(see Known Limitations)* |
| datetime_columns_present | [] *(see Known Limitations)* |
| listed_time_parse_rate | null *(see Known Limitations)* |

**Top 20 skills by frequency (from fixture):**

| Rank | Skill | Count |
|------|-------|-------|
| 1 | Python | 3 |
| 2 | Docker | 3 |
| 3 | Machine Learning | 2 |
| 4–21 | (all other skills) | 1 each |

---

## Task Assessment

| Task | Status | Notes |
|------|--------|-------|
| 1. Create `reports/m1_real_data_smoke_report.md` | **PASS** | This file |
| 2. Add `src/jobsrec/data/profile.py` | **PASS** | Fully implemented |
| 3. Profiler summarises all required metrics | **PASS** | 13-key output |
| 4. `profile-silver` CLI command | **PASS** | Writes JSON + console summary |
| 5. `tests/fixtures/kaggle_minimal/` with Kaggle layout | **PASS** | 3 CSVs, realistic structure |
| 6. All 6 named tests present | **PASS** | Plus 27 additional assertions |
| 7. `python -m pytest -q` passes | **PASS** | 86 passed, 0 failed |
| 8. End-to-end CLI smoke test | **PASS** | All 4 commands succeed |
| 9. README updated with M1 section | **PASS** | Workflow + new files documented |
| 10. This smoke report | **PASS** | PASS for all tasks |

---

## Known Limitations

### 1. listed_time / salary columns not in silver output
The current `build_silver` pipeline preserves only `POSTINGS_OPTIONAL` columns
(`formatted_experience_level`, `formatted_work_type`, `location`).
Columns like `listed_time`, `salary_min`, `salary_max`, `pay_period` from
`postings.csv` are **not** forwarded to the silver Parquet, so the profiler
correctly reports `null` / `[]` for these categories.

**Impact:** The profiler can analyse these columns when the real Kaggle silver
dataset includes them (if `POSTINGS_OPTIONAL` is extended in a future milestone).
The profiler code already handles them — it simply finds no matching columns in
the current silver schema.

**Workaround for M2:** Extend `POSTINGS_OPTIONAL` in `schema.py` to include
`listed_time`, `salary_min`, `salary_max`, `pay_period` and re-run `build-silver`.

### 2. Fixture size (10 postings)
The synthetic fixture uses only 10 postings to keep tests fast and avoid
committing real Kaggle data. Skill frequency statistics are not meaningful
at this scale.

### 3. No real Kaggle data in CI
Full validation against the 124K-posting Kaggle dataset requires manual download
(`kaggle datasets download -d arshkon/linkedin-job-postings`). This is by design.
