# Data Contract — Milestone 1.5

This document is the authoritative schema reference for every file produced or
consumed by the `jobsrec` pipeline.

> **Milestone history**
> - Milestone 0: initial schema draft.
> - Milestone 1: build-silver, build-tfidf, profile-silver, recommend pipeline.
> - Milestone 1.5: expanded optional columns, salary join, richer profiling.

---

## 1. Input files (raw / Kaggle)

### 1.1 `postings.csv`

#### Required columns

| Column | Type | Notes |
|--------|------|-------|
| `job_id` | int64 | Unique job identifier |
| `title` | str | Job title |
| `description` | str | Full job description text |

Validation rule: `ValueError` is raised if any required column is absent.

#### Optional columns (preserved when present)

All columns below are carried into the silver dataset unchanged when they
exist in `postings.csv`.  Missing optional columns are **silently skipped**
(no back-filling with empty strings at the silver layer).

| Column | Type | Category | Notes |
|--------|------|----------|-------|
| `listed_time` | float / int | temporal | Unix timestamp in ms |
| `original_listed_time` | float / int | temporal | Unix timestamp in ms |
| `expiry` | float / int | temporal | Unix timestamp in ms |
| `closed_time` | float / int | temporal | Unix timestamp in ms |
| `formatted_work_type` | str | job metadata | e.g. "Full-time" |
| `formatted_experience_level` | str | job metadata | e.g. "Mid-Senior level" |
| `work_type` | str | job metadata | e.g. "FULL_TIME" |
| `location` | str | job metadata | e.g. "San Francisco, CA" |
| `remote_allowed` | str / int | job metadata | 1 = remote, 0 = on-site |
| `skills_desc` | str | job metadata | Free-text skills description |
| `sponsored` | str / int | job metadata | 1 = sponsored listing |
| `normalized_salary` | float | compensation | Normalised annual salary |
| `application_url` | str | application | URL for external application |
| `application_type` | str | application | e.g. "OffSite" |
| `views` | int | engagement | Total posting view count |
| `applies` | int | engagement | Total application count |
| `company_id` | int | company | FK → companies/companies.csv |
| `zip_code` | str | geo | US ZIP code |
| `fips` | str | geo | FIPS county code |

> **Temporal column preservation policy**: temporal columns are kept in
> their raw numeric form (Unix-ms integer / float).  They are **not** cast
> to `datetime` at the silver layer.  The `profile-silver` command reports
> parse-success rates for `listed_time`, `expiry`, and `closed_time`.

### 1.2 `jobs/job_skills.csv`

| Column | Type | Required | Notes |
|--------|------|----------|-------|
| `job_id` | int64 | ✅ | Foreign key → `postings.job_id` |
| `skill_abr` | str | ✅ | Abbreviated skill code |

### 1.3 `mappings/skills.csv`

| Column | Type | Required | Notes |
|--------|------|----------|-------|
| `skill_abr` | str | ✅ | Primary key |
| `skill_name` | str | ✅ | Human-readable skill label |

### 1.4 `jobs/salaries.csv` (optional)

The salary join is attempted whenever this file exists.  The pipeline
continues without error when the file is absent.

| Column | Type | Required | Notes |
|--------|------|----------|-------|
| `salary_id` | int | optional | Row identifier (not used) |
| `job_id` | int64 | ✅ | FK → `postings.job_id` |
| `min_salary` | float | optional | Minimum salary |
| `max_salary` | float | optional | Maximum salary |
| `med_salary` | float | optional | Median salary |
| `pay_period` | str | optional | e.g. "YEARLY", "HOURLY" |
| `currency` | str | optional | e.g. "USD" |
| `compensation_type` | str | optional | e.g. "BASE_SALARY" |

**Aggregation rule** (multiple rows per `job_id`):
- Numeric fields (`min_salary`, `max_salary`, `med_salary`): **max** of
  non-null values.
- String fields (`pay_period`, `currency`, `compensation_type`): **first
  non-null, non-empty** value.

This ensures exactly one salary row per job after aggregation, so the
left-join with `postings` never introduces duplicate job rows.

---

## 2. Silver dataset — `data/silver/jobs.parquet`

Produced by `python -m jobsrec.cli build-silver`. Single Parquet file,
not partitioned.

### 2.1 Required columns

| Column | Type | Nullable | Notes |
|--------|------|----------|-------|
| `job_id` | int64 | ❌ | Unique per row |
| `title` | str | ❌ | |
| `description` | str | ❌ | |
| `skills_text` | str | ✅ | Comma-joined skill names; `""` if none |
| `job_card_text` | str | ❌ | Deterministic concatenation — see §3 |

### 2.2 Optional columns (present when source data has them)

All optional postings columns from §1.1 may appear, plus the salary columns
from §1.4.  Their presence is recorded in `manifest.json` (see §5).

> **Jobs without skills are preserved**: `skills_text = ""` for jobs with
> no matching rows in `job_skills.csv`.  Such jobs are counted by
> `profile-silver` under `n_jobs_without_skills`.

---

## 3. `job_card_text` format

Sections are always emitted in this exact order; missing optional fields are
omitted entirely (not replaced with a placeholder string).

```
Title: {title}
Experience: {formatted_experience_level}   ← omitted if blank
Work type: {formatted_work_type}           ← omitted if blank
Location: {location}                       ← omitted if blank
Skills: {skills_text}                      ← omitted if blank
Description: {description}
```

Newline separator: `\n` (Unix).  Leading / trailing whitespace stripped per field.

---

## 4. Gold artifacts — `data/gold/`

| File | Format | Notes |
|------|--------|-------|
| `tfidf_vectorizer.joblib` | joblib | Fitted `TfidfVectorizer` |
| `tfidf_matrix.npz` | scipy sparse | Shape `(n_jobs, vocab)` |
| `job_index.parquet` | Parquet | `job_id`, `job_card_text` ordered by matrix row |
| `manifest.json` | JSON | See §5 |

---

## 5. Manifest schema

Every pipeline stage writes a `manifest.json` alongside its outputs.

### `build-silver` manifest

```json
{
  "stage": "build-silver",
  "created_at": "2025-01-01T00:00:00+00:00",
  "input_rows": 123456,
  "output_rows": 118000,
  "input_path": "data",
  "output_path": "data/silver/jobs.parquet",
  "config": {},
  "preserved_optional_columns": [
    "listed_time", "original_listed_time", "expiry", "closed_time",
    "formatted_work_type", "formatted_experience_level", "work_type",
    "location", "remote_allowed", "skills_desc", "sponsored",
    "normalized_salary", "application_url", "application_type",
    "views", "applies", "company_id", "zip_code", "fips"
  ],
  "joined_optional_tables": ["jobs/salaries.csv"],
  "salary_columns_added": [
    "min_salary", "max_salary", "med_salary", "pay_period", "currency",
    "compensation_type"
  ]
}
```

---

## 6. Deprecation / versioning

- Breaking schema changes increment the `version` field in `manifest.json`.
- Additive changes (new nullable columns) are backwards compatible.
