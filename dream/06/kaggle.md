# Kaggle LinkedIn Dataset — Synthesis

Source: `arshkon/linkedin-job-postings` (Kaggle)
Files: `postings.csv`, `jobs/job_skills.csv`, `jobs/salaries.csv`, `mappings/skills.csv`

---

## Shape

- **123,849 US LinkedIn job postings**, English, all collected in a narrow window.
- Temporal reality: `listed_time` covers 2 months — March 2024 has **1 row**, April has 123,848.
  This makes the dataset a **point-in-time snapshot**, not a time series.
- Other time columns: `original_listed_time` (5 months), `expiry` (7 months — lifecycle, not demand), `closed_time` (sparse, 1,073 rows).
- Scale saturates early: centroid distance at 10k was 0.1914, at 100k was 0.1701. The signal difference was small.

## Skills

Skills were **pre-labeled by LinkedIn**, joined from `job_skills.csv → skills.csv`.
- Only 35 coarse categories: `Information Technology`, `Sales`, `Manufacturing`, `Health Care Provider`, etc.
- 98.6% coverage (1,753 / 123,849 jobs had no skills — preserved via left join).
- Top: IT > Sales > Management > Manufacturing > Health Care Provider > Business Development.
- Skills were **not** extracted from job text; they came from LinkedIn's own ontology.
  This means no NLP extraction quality problem, but also no signal about what the posting *says*.

## Clusters (TF-IDF baseline)

8 distinct semantic clusters visible in TF-IDF/SVD space:
- Receptionist / Sales / Customer Service
- Healthcare (Registered Nurse, Medical Assistant)
- Retail / Store Manager
- Admin / Accounting / Logistics
- Finance / PM / Accounting (white-collar mixed)
- Maintenance / Technician / Automotive
- Sales leadership (small, niche)
- Long-term care / Nursing facility (small, niche)

Mean cosine similarity between pairs: 0.54 — decent sector separation with just TF-IDF.
Cluster structure was visible and stable; data/marketing/business was the only visibly growing share within the April window.

## Salary

- Separate `salaries.csv` with **many rows per job_id** — required deterministic aggregation before join.
  Rule: take `max()` for numeric bounds, `first non-null` for strings.
- `min_salary / max_salary`: 29,793 / 123,849 (~24% coverage).
- `normalized_salary`: 100% coverage (computed from raw values, not independently verified).
- `pay_period / currency`: ~29% coverage.
- Salary columns existed in **both** `postings.csv` and `salaries.csv` — had to drop the `postings.csv`
  versions before merge or suffer `_x / _y` suffix pollution.

## Metadata Distribution

Work types: Full-time 79.8%, Contract 9.8%, Part-time 7.8%, Internship/Temp/Other small.
Experience: Mid-Senior 33.5%, Entry 29.6%, Associate 7.9%, Director/Exec ~4%.
Remote: 15,246 postings (~12.3%) explicitly marked remote.
Geography: US only. Top cities: New York, Houston, Chicago, Boston, Atlanta, Phoenix.

## Problems Solved

| Problem | Solution |
|---------|----------|
| `listed_time` stored as Unix epoch milliseconds (not datetime) | `pd.to_datetime(unit="ms")` after detecting values < 1990 with standard parse |
| Many-to-one salary rows exploding join | Aggregate before merge: `max()` numeric, `first()` string |
| Overlapping column names between `postings.csv` and `salaries.csv` | Drop salary cols from postings before merge |
| Jobs with no skill assignment would be dropped | Left join + `fillna("")` on `skills_text` |
| Temporal analysis producing meaningless "trends" | `temporal-audit` gate: check month count and support before any drift claim |
| 10k vs 100k accuracy question | Centroid distance converged — 10k sufficient for baseline checks |

## What This Dataset Cannot Do

- Real temporal trend analysis — 2 months, 1 of which has 1 row.
- Skill extraction quality measurement — skills were pre-labeled, not extracted from text.
- Latin American / Spanish-language market signals — US English only.
- Fine-grained skill vocabulary — 35 coarse LinkedIn categories, no tool names, no tech stack.

## Why We Moved Away

Our target market is Latin America (Argentina-first). The Kaggle dataset is US-only and English-only.
Skills are a LinkedIn taxonomy, not extracted from unstructured HTML — so there's nothing to learn about
extraction quality or skill coverage from it.
The temporal coverage is useless for anything beyond a two-bucket demo.
We built our own scraper to get fresh Argentine LinkedIn postings in Spanish with raw HTML.

## What Carried Forward

- **Bronze → Silver → Gold pipeline pattern** (SQLite → Parquet → index).
- **TF-IDF baseline** as a fast, local, interpretable sanity check before any embedding work.
- **Temporal audit as a mandatory first gate** — never claim "market evolution" without checking month count and per-month support.
- **Left join for skill enrichment** — preserve zero-skill jobs rather than silently dropping them.
- **Epoch millisecond timestamp handling** — scraped jobs likely carry timestamps in a similar format.
- **Salary aggregation pattern** — if future scraped data includes multi-row compensation tables, same rule applies.
- The lesson that **signal saturates early**: 10k is enough for embedding structure checks, 100k adds little.
