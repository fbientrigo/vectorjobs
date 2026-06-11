# M2.3 Temporal Comparison / Drift Demo

This prototype builds a fast, meeting-ready temporal comparison view over the
silver jobs dataset without LLM calls, FAISS, or notebook dependencies.

With the current `data/silver/jobs.parquet`, the available `listed_time` range
is only 2024-03 to 2024-04. The current full-data monthly support is:

| month | rows |
| --- | ---: |
| 2024-03 | 1 |
| 2024-04 | 123,848 |

The reliability label for the current dataset is `limited_temporal_coverage`.
Outputs must be described as a two-month temporal comparison / drift demo, not
as stable trends, market evolution, seasonality, or durable skill movement.

## Audit Command

```bash
python -m jobsrec.cli temporal-audit \
    --input data/silver/jobs.parquet \
    --output-dir reports/temporal_audit
```

The audit writes:

- `report.md`
- `summary.json`
- `monthly_counts.parquet`
- `weekly_counts.parquet`
- `daily_counts.parquet`
- `temporal_column_coverage.parquet`

Required audit warnings:

- `< 6 months` means limited temporal coverage.
- Any month with fewer than 1,000 rows is low support.
- A first/last comparison is likely noisy when either endpoint month is low support.
- Two months are a two-bucket comparison, not a trend.
- `original_listed_time` can be used as a wider but sparse comparison basis.
- `expiry` is lifecycle/expiration timing, not posting-demand coverage.

## TF-IDF/SVD Demo Command

```bash
python -m jobsrec.cli temporal-demo \
    --input data/silver/jobs.parquet \
    --output-dir reports/temporal_tfidf_10k \
    --sample-size 10000 \
    --representation tfidf_svd \
    --time-column listed_time \
    --sampling-mode temporal-stride
```

Wider-but-sparse original listing basis:

```bash
python -m jobsrec.cli temporal-demo \
    --input data/silver/jobs.parquet \
    --output-dir reports/temporal_original_time_tfidf_10k \
    --sample-size 10000 \
    --representation tfidf_svd \
    --time-column original_listed_time \
    --sampling-mode temporal-stride
```

Legacy explicit output paths remain supported:

```bash
python -m jobsrec.cli temporal-demo \
    --silver-path data/silver/jobs.parquet \
    --output-dir data/gold/trends_10k \
    --figures-dir reports/figures_10k \
    --report-path reports/m2_2_temporal_demo_10k.md \
    --sample-size 10000 \
    --sampling-mode temporal-stride \
    --representation tfidf_svd \
    --time-column listed_time \
    --centroid-weighting unweighted
```

## Safe Semantic Smoke Command

```bash
python -m jobsrec.cli temporal-demo \
    --input data/silver/jobs.parquet \
    --sample-size 1000 \
    --representation semantic_embeddings \
    --embedding-backend mock \
    --embedding-model deterministic-mock \
    --embedding-batch-size 16 \
    --device cpu \
    --embedding-cache-dir data/cache/embeddings \
    --output-dir reports/temporal_semantic_mock_1k
```

The mock backend is deterministic and does not download model weights. It is
the default safe path for unit tests and smoke validation.

## Salary-Weighted Centroid Command

```bash
python -m jobsrec.cli temporal-demo \
    --input data/silver/jobs.parquet \
    --sample-size 10000 \
    --representation tfidf_svd \
    --time-column original_listed_time \
    --centroid-weighting both \
    --output-dir reports/temporal_salary_weighted_tfidf_10k
```

Salary-weighted centroids are an additional view beside the unweighted
baseline. They describe only the salary-disclosed USD subset.

## Method

- Parse the selected `--time-column` with `pandas.to_datetime(errors="coerce")`,
  with numeric epoch fallback when plain parsing lands before 1990.
- Bucket valid timestamps by `YYYY-MM`.
- Sample with temporal stride by default to preserve first-to-last month
  coverage and at least one row per available month when `sample_size` allows.
- For `tfidf_svd`, vectorize `job_card_text` with TF-IDF and apply
  TruncatedSVD when the matrix is large enough.
- For `semantic_embeddings`, use an explicit backend, batch size, device, cache
  directory, and row cap.
- Compute normalized monthly centroids and consecutive-month distance as
  `1.0 - cosine_similarity`.
- If `--centroid-weighting both|salary` is enabled, compute salary-weighted
  centroids using `log1p(annual_salary)`, normalized within month and clipped to
  `[0.25, 4.0]`.
- Compare `skills_text` shares between the first and last selected month.

## Interpretation Rules

- Use "temporal comparison" or "temporal drift demo".
- Do not use "market evolution", "stable trend", or "seasonality" for the
  current two-month dataset.
- Treat positive and negative skill-share deltas as bucket differences, not as
  rising or declining skills over time.
- Treat salary-weighted centroid drift as a salary-disclosed subset view, not a
  full-market estimate.
- Treat the historical figure names `top_rising_skills.png` and
  `top_declining_skills.png` as positive and negative skill-share delta plots.
- Surface warnings in markdown reports and manifests, not only logs.

## Required Outputs

- `monthly_centroid_drift.parquet`
- `skill_growth.parquet`
- `temporal_manifest.json`
- markdown report
- `job_volume_by_month.png`
- `centroid_drift_by_month.png`
- `top_rising_skills.png`
- `top_declining_skills.png`

For semantic embedding runs, also write:

- consecutive-month drift columns: `month_from`, `month_to`, `n_from`, `n_to`,
  `representation`, `embedding_backend`, `embedding_model`,
  `cosine_similarity`, `cosine_distance`
- `monthly_centroid_metadata.parquet`
- centroid storage path, currently `monthly_centroids.npy`
- embedding cache provenance in the manifest

For salary-weighted runs, also write:

- `monthly_centroid_drift_salary_weighted.parquet`
- `monthly_centroid_metadata_salary_weighted.parquet`
- `monthly_centroids_salary_weighted.npy`
- `salary_weight_diagnostics.parquet`
- manifest fields for `centroid_weighting`, `salary_weight_strategy`,
  `salary_rows_used`, `salary_coverage`, `salary_currency_filter`, and clipping
  limits

Required salary-weighted drift columns:

- `month_from`
- `month_to`
- `n_from`
- `n_to`
- `n_salary_from`
- `n_salary_to`
- `salary_coverage_from`
- `salary_coverage_to`
- `representation`
- `time_column`
- `centroid_weighting`
- `cosine_similarity`
- `cosine_distance`

Optional clustering and similarity-sample artifacts are exploratory only.

## Current Scope Boundaries

- No FAISS in this milestone. The demo computes aggregate centroids and small
  smoke-test similarities, not large-scale approximate nearest-neighbor search.
- No automatic large model downloads.
- Real Qwen3 runs must be explicit, small, cached, and batch-controlled.
- Bootstrap stability is deferred until a small explicit seed-list design is
  added.
