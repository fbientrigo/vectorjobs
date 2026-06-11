# Temporal Audit

## Command
`python -m jobsrec.cli temporal-audit --input data\silver\jobs.parquet --output-dir reports\temporal_audit --time-column listed_time`

## Summary
- Input path: `data\silver\jobs.parquet`
- Primary time column: `listed_time` (canonical posting-listing time)
- Total rows: 123849
- Valid selected-time rows: 123849
- Invalid/missing selected-time rows: 0
- Parse success rate: 1.0000
- Min date: 2024-03-24T21:50:14
- Max date: 2024-04-20T00:26:56
- Number of months: 2
- Reliability label: `limited_temporal_coverage`

## Warnings
- Only 2 months are available; this is a temporal comparison, not a stable trend.
- Only 2 months are available; this is a two-bucket comparison, not a trend.
- Months below 1000 rows may make first/last or consecutive comparisons noisy: 2024-03=1
- First/last comparison is likely noisy because an endpoint month has low support.
- Available months < 6; temporal outputs should be labeled limited.

## Monthly Counts
| month | rows | job_card_text_coverage | skills_text_coverage |
| --- | --- | --- | --- |
| 2024-03 | 1 | 1.0 | 1.0 |
| 2024-04 | 123848 | 1.0 | 0.9858455526128803 |

## Temporal Column Coverage
| time_column | interpretation | valid_rows | number_of_months | reliability_label |
| --- | --- | --- | --- | --- |
| listed_time | canonical posting-listing time | 123849 | 2 | limited_temporal_coverage |
| original_listed_time | wider but sparse original listing time | 123849 | 5 | limited_temporal_coverage |
| expiry | expiration lifecycle time, not posting-demand coverage | 123849 | 7 | sufficient_temporal_coverage |
| closed_time | posting close lifecycle time, sparse when present | 1073 | 1 | demo_only |

## Interpretation
This audit describes temporal coverage only. It does not establish market evolution.

## Deferred Tasks
- Bootstrap stability for drift and skill growth is deferred; add a small explicit seed list before using confidence intervals.
