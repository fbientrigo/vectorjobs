# M2.2 Temporal Drift Demo: 10k Two-Month Comparison

## Command

`python -m jobsrec.cli temporal-demo --silver-path data\silver\jobs.parquet --output-dir data\gold\trends_10k --figures-dir reports\figures_10k --report-path reports\m2_2_temporal_demo_10k.md --sample-size 10000 --sampling-mode temporal-stride`

## Run Summary

- Input path: `data\silver\jobs.parquet`
- Rows input: 123,849
- Rows selected: 10,000
- `listed_time` parse success rate: 1.0000
- Sampling mode: temporal-stride
- Month range: 2024-03 to 2024-04
- Months covered: 2
- Reliability label: `limited_temporal_coverage`

## Reliability Caveats

- This output is a two-month temporal comparison / drift demo.
- It is not a stable trend analysis and should not be described as market evolution.
- The current full dataset has only 1 March row and 123,848 April rows.
- The sampled 10k run has 1 March row and 9,999 April rows.
- First-vs-last month comparisons are noisy because the first month has extremely low support.
- `< 6 months` means this dataset has limited temporal coverage.

## March-to-April Centroid Distance

| month_to | month_from | cosine_distance | jobs_in_month |
| --- | --- | --- | --- |
| 2024-04 | 2024-03 | 0.19140330082871493 | 9999 |

## Largest Positive Skill-Share Deltas

These are first-vs-last bucket deltas, not evidence that skills are rising over time.

| skill | first_share | last_share | share_delta |
| --- | --- | --- | --- |
| information technology | 0.0 | 0.11970767651144532 | 0.11970767651144532 |
| sales | 0.0 | 0.10183004167421635 | 0.10183004167421635 |
| health care provider | 0.0 | 0.08189889472730567 | 0.08189889472730567 |
| business development | 0.0 | 0.0640212598900767 | 0.0640212598900767 |
| engineering | 0.0 | 0.06045781240562904 | 0.06045781240562904 |
| other | 0.0 | 0.058041915806003506 | 0.058041915806003506 |
| finance | 0.0 | 0.03750679470918645 | 0.03750679470918645 |
| marketing | 0.0 | 0.025850093615993237 | 0.025850093615993237 |
| administrative | 0.0 | 0.02488373497614302 | 0.02488373497614302 |
| accounting/auditing | 0.0 | 0.02301141511143323 | 0.02301141511143323 |
| customer service | 0.0 | 0.02047472368182642 | 0.02047472368182642 |
| project management | 0.0 | 0.01775684000724769 | 0.01775684000724769 |
| analyst | 0.0 | 0.0174548529322945 | 0.0174548529322945 |
| research | 0.0 | 0.014072597692818748 | 0.014072597692818748 |
| human resources | 0.0 | 0.012683457148034065 | 0.012683457148034065 |

## Largest Negative Skill-Share Deltas

These are first-vs-last bucket deltas, not evidence that skills are declining over time.

| skill | first_share | last_share | share_delta |
| --- | --- | --- | --- |
| manufacturing | 0.5 | 0.08805943105635079 | -0.4119405689436492 |
| management | 0.5 | 0.10080328561937549 | -0.39919671438062454 |

## Figures

- `reports\figures_10k\job_volume_by_month.png`
- `reports\figures_10k\centroid_drift_by_month.png`
- `reports\figures_10k\top_rising_skills.png`
- `reports\figures_10k\top_declining_skills.png`
- `reports\figures_10k\job_cluster_map_svd.png`
- `reports\figures_10k\similarity_distribution.png`

The figure filenames are historical. Interpret `top_rising_skills.png` and `top_declining_skills.png` as positive and negative skill-share deltas between the two buckets.

## Optional Clustering Summary

The cluster map is exploratory. It is useful for checking whether the representation has structure, but it is not a temporal evaluation metric.

| cluster_id | n_jobs | top_titles |
| --- | --- | --- |
| 0 | 1056 | ['Patient Care Technician', 'Registered Nurse', 'Registered Nurse - RN - LTAC', 'Nurse - LPN - LTC', 'HOSPICE REGISTERED NURSE'] |
| 1 | 1043 | ['Senior Software Engineer', 'Data Engineer', 'Manufacturing Engineer', 'Software Engineer', 'Electrical Engineer'] |
| 2 | 1268 | ['Maintenance Technician', 'Salesperson', 'Warehouse Associate', 'Delivery Specialist', 'Service Technician'] |
| 3 | 48 | ['Sales Manager', 'Business Development Manager'] |
| 4 | 2119 | ['Project Manager', 'Account Manager', 'Business Development Manager', 'Financial Analyst', 'Senior Project Manager'] |
| 5 | 451 | ['Retail Sales Associate', 'Store Manager', 'ASSISTANT STORE MANAGER', 'Customer Service Representative', 'OPERATIONS ASSISTANT MANAGER'] |
| 6 | 1970 | ['Administrative Assistant', 'Sales Executive', 'Account Executive', 'Mortgage Loan Officer', 'Accounting Manager'] |
| 7 | 2045 | ['Receptionist', 'Customer Service Representative', 'Auto Glass Installation Technician Trainee', 'Retail Sales and Store Support', 'Sales Associate'] |

## Optional Similarity Summary

- Jobs sampled: 500
- Pair similarities: 249,500
- Mean similarity: 0.5339
- Median similarity: 0.5598

## Known Limitations

- TF-IDF/SVD is a fast baseline, not a semantic embedding model.
- Numeric `listed_time` values are interpreted as epoch milliseconds or seconds when plain pandas parsing lands before 1990.
- Skill-share comparison depends on the quality and consistency of `skills_text`.
- Month-to-month drift is noisy when a compared month has few postings.
- This dataset has limited temporal coverage and should not be used to claim stable evolution.

## Next Step

Use `temporal-audit` to make coverage warnings visible before running larger demos. Keep this run as a TF-IDF/SVD temporal drift baseline; use the mock semantic embedding path for schema-safe smoke tests before any explicit small Qwen3 run.
