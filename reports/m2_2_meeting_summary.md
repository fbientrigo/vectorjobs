# M2.2 Meeting Summary: Two-Month Temporal Drift Demo

## Project Objective

Build a fast, meeting-ready temporal analytics proof of concept over LinkedIn job postings using the existing `jobsrec` silver dataset and lightweight local computation. With the current data, the correct interpretation is a two-month temporal comparison / drift demo, not evidence of stable market evolution.

## What Was Built

- A `temporal-demo` CLI command in `jobsrec` that reads `data/silver/jobs.parquet`.
- Temporal-stride sampling for 10k and 100k runs, preserving first-to-last date coverage.
- TF-IDF with optional TruncatedSVD as the baseline text representation over `job_card_text`.
- Monthly centroid drift using normalized monthly vectors and consecutive-month cosine distance.
- First-vs-last month skill-share comparison from `skills_text`.
- Markdown reports, manifests, parquet outputs, and PNG figures for 10k and 100k demos.

## Dataset Coverage

- Silver input rows: 123,849.
- Valid `listed_time` rows: 123,849, with 100% parse success after numeric epoch handling.
- Month range covered: 2024-03 to 2024-04.
- Number of months covered: 2.
- Full-data monthly support: 2024-03 has 1 row; 2024-04 has 123,848 rows.
- 10k demo rows selected: 10,000.
- 100k demo rows selected: 100,000.
- Reliability label: `limited_temporal_coverage`.

## Method

The demo represents each posting's `job_card_text` with TF-IDF/SVD, groups valid postings into monthly buckets, computes one normalized centroid per month, and reports centroid distance as `1.0 - cosine_similarity` between the March and April centroids. Skill-share output compares normalized `skills_text` tokens between the first selected month and the last selected month.

Because there are only two available months and March has extremely low support, these outputs are descriptive diagnostics for the current file. They should not be described as trends, stable evolution, or market movement.

## Descriptive Results

- 10k March-to-April centroid distance: 0.1914.
- 100k March-to-April centroid distance: 0.1701.
- 100k largest positive skill-share deltas: information technology, sales, health care provider, business development, engineering.
- 100k largest negative skill-share deltas: manufacturing, management.
- The 100k run completed in 195.36 seconds and produced the required report, manifest, parquet outputs, and figures.

These values describe the two selected month buckets only. The skill-share deltas are not evidence that skills are rising or declining over time.

## Figure List

- `reports/figures_10k/job_volume_by_month.png`
- `reports/figures_10k/centroid_drift_by_month.png`
- `reports/figures_10k/top_rising_skills.png`
- `reports/figures_10k/top_declining_skills.png`
- `reports/figures_10k/job_cluster_map_svd.png`
- `reports/figures_10k/similarity_distribution.png`
- `reports/figures_100k/job_volume_by_month.png`
- `reports/figures_100k/centroid_drift_by_month.png`
- `reports/figures_100k/top_rising_skills.png`
- `reports/figures_100k/top_declining_skills.png`
- `reports/figures_100k/job_cluster_map_svd.png`
- `reports/figures_100k/similarity_distribution.png`

The historical figure filenames still contain `top_rising_skills` and `top_declining_skills`, but the current interpretation should be positive and negative first-vs-last skill-share deltas.

## Demo Talking Points

- The pipeline is fully local and fast enough for iteration: no LLM calls, no FAISS, no notebook dependency.
- Temporal-stride sampling keeps chronological coverage while making 10k and 100k demo runs practical.
- Monthly centroid distance gives a simple, explainable signal for how March and April posting text differ in this file.
- Skill-share comparison turns raw `skills_text` into a readable March-vs-April bucket comparison.
- The 100k result is close to the 10k result, but both are constrained by the same two-month coverage and low March support.

## Limitations

- The current silver file spans only two months: 2024-03 and 2024-04.
- The March bucket has only 1 full-data row, so March-vs-April comparisons are noisy.
- `< 6 months` means the dataset has limited temporal coverage.
- Two months are not enough to claim a stable trend, evolution, seasonality, or market shift.
- TF-IDF/SVD is an interpretable baseline, not a semantic embedding model.
- Skill-share quality depends on how consistently `skills_text` is populated and normalized.
- Optional cluster and similarity figures are exploratory and should not be treated as evaluation metrics.

## Next Technical Step

Keep the next step inside the current repo scope: use `temporal-audit` as the front door for temporal work, preserve the TF-IDF/SVD baseline, and use the deterministic mock semantic embedding path for safe smoke tests. Run a small explicit Qwen3 smoke only if the local machine can tolerate it. Do not add FAISS until there is a retrieval/indexing milestone that needs approximate nearest-neighbor search.

## Already Validated By Tests

- Month parsing and month bucket creation.
- Numeric epoch timestamp handling.
- Invalid dates excluded from temporal calculations.
- Temporal-stride sampling covers first and last month on synthetic data.
- Random sampling is deterministic with `random_state=42`.
- Centroid drift output has required columns.
- Skill-share comparison detects positive and negative deltas.
- CLI writes report, manifest, parquet outputs, and required figures on tiny synthetic data.
- Temporal audit output schema.
- Reliability labels and low-support warnings.
- Deterministic mock semantic embedding path.
- Semantic centroid drift output schema.
- Full repository test suite passed: 149 tests.
