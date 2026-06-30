# Data Contract

This is the current contract for the `jobsrec` pipeline.

## Bronze SQLite

Input is the official scraper database:

```bash
python -m jobsrec.cli build-silver --input-db data/bronze/jobs.db --output-dir data/silver
```

Required tables:

| Table | Required columns |
| --- | --- |
| `jobs` | `id`, `title`, `company`, `location`, `description`, `status` |
| `job_observations` | `job_id`, `crawl_id`, `seen_at` |
| `crawl_runs` | `id`, `started_at` |

`jobs.id` and `job_observations.job_id` are text IDs. Do not cast them to integers.

## Silver Parquet

Schema version: **0.2.0**

Output is a single file: `data/silver/jobs.parquet`.

Required columns:

| Column | Type | Notes |
| --- | --- | --- |
| `job_id` | string | From `jobs.id` — always string, never cast to int |
| `title` | string | Trimmed |
| `company_name` | string | From `company['denominacion']` (falls back to `nombre`, `name`) |
| `company_confidential` | boolean/null | From `company['confidencial']` |
| `company_raw` | string | Original `company` field text |
| `company_city` | string | From `company['ciudad']`; `""` when absent |
| `company_region` | string | From `company['provincia']`; `""` when absent |
| `company_industry` | string | From `company['industria']`; `""` when absent |
| `company_parse_error` | bool | True if `ast.literal_eval` failed on `company` |
| `location` | string/null | Nullable; current official DB has all nulls |
| `description_html` | string | Original scraper HTML |
| `description_text` | string | HTML stripped to readable text |
| `status` | string | Scraper status, including deleted rows |
| `first_seen_at` | string/null | Minimum observation `seen_at` |
| `last_seen_at` | string/null | Maximum observation `seen_at` |
| `times_seen` | int | Observation row count |
| `crawl_count` | int | Distinct crawl count |
| `skills_text` | string | Always `""` until bronze has a skills table |
| `job_card_text` | string | Deterministic retrieval text |

Rows with both blank title and blank description are dropped. Deleted rows are kept when they still have useful title or description content.

**Note on company field:** The scraper stores `company` as a Python `repr()` dict string, parsed with `ast.literal_eval`. Non-confidential companies use `denominacion` as the name key.

## `job_card_text`

Sections are emitted in order, omitting blank sections:

```text
Title: {title}
Location: {location}
Skills: {skills_text}
Description: {description_text}
```

## Manifest

`data/silver/manifest.json` records:

- `bronze_format: sqlite`
- `silver_schema_version`: e.g. `"0.2.0"`
- input/output row counts
- source table counts
- exact silver columns
- `skills_source: none` until the scraper provides skills

## Extraction Candidates

Output: `data/silver/job_extraction_candidates.parquet`
Schema version: **extraction_v0.1**

One row per extracted text unit (title, paragraph, or `<li>` item) per job.

| Column | Type | Notes |
| --- | --- | --- |
| `job_id` | string | FK → silver `jobs.parquet` |
| `candidate_index` | int32 | 0-based index within job |
| `candidate_text` | string | Extracted and cleaned text |
| `candidate_source` | string | `"title"`, `"paragraph"`, or `"li"` |
| `section_name` | string | Detected Spanish section anchor or `""` |
| `skills_regex_raw` | string | JSON-encoded `list[str]` of raw matched strings |
| `skills_normalized` | string | JSON-encoded `list[str]` of canonical skill names |

`skills_regex_raw` and `skills_normalized` are always valid JSON arrays (use `json.loads()` to decode). Skill dictionary version is recorded in `extraction_manifest.json`.

Companion: `data/silver/extraction_manifest.json` — records schema version, run timestamp, counts, top skills, and skill dict version.
