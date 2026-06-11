# M2.2 Temporal Drift Demo: 100k Two-Month Comparison

## Command

`python -m jobsrec.cli temporal-demo --silver-path data\silver\jobs.parquet --output-dir data\gold\trends_100k --figures-dir reports\figures_100k --report-path reports\m2_2_temporal_demo_100k.md --sample-size 100000 --sampling-mode temporal-stride`

## Run Summary

- Input path: `data\silver\jobs.parquet`
- Rows input: 123,849
- Rows selected: 100,000
- `listed_time` parse success rate: 1.0000
- Sampling mode: temporal-stride
- Month range: 2024-03 to 2024-04
- Months covered: 2
- Reliability label: `limited_temporal_coverage`
- Runtime seconds: 195.36

## Reliability Caveats

- This output is a two-month temporal comparison / drift demo.
- It is not a stable trend analysis and should not be described as market evolution.
- The current full dataset has only 1 March row and 123,848 April rows.
- The sampled 100k run has 1 March row and 99,999 April rows.
- First-vs-last month comparisons are noisy because the first month has extremely low support.
- `< 6 months` means this dataset has limited temporal coverage.

## March-to-April Centroid Distance

| month_to | month_from | cosine_distance | jobs_in_month |
| --- | --- | --- | --- |
| 2024-04 | 2024-03 | 0.17008681042637275 | 99999 |

## Largest Positive Skill-Share Deltas

These are first-vs-last bucket deltas, not evidence that skills are rising over time.

| skill | first_share | last_share | share_delta |
| --- | --- | --- | --- |
| information technology | 0.0 | 0.122078047426872 | 0.122078047426872 |
| sales | 0.0 | 0.10471674780269166 | 0.10471674780269166 |
| health care provider | 0.0 | 0.08338008457450338 | 0.08338008457450338 |
| business development | 0.0 | 0.06587400691315128 | 0.06587400691315128 |
| engineering | 0.0 | 0.06046292777385671 | 0.06046292777385671 |
| other | 0.0 | 0.05930470347648262 | 0.05930470347648262 |
| finance | 0.0 | 0.03862557383378275 | 0.03862557383378275 |
| marketing | 0.0 | 0.02501040592142172 | 0.02501040592142172 |
| accounting/auditing | 0.0 | 0.02372550084152234 | 0.02372550084152234 |
| administrative | 0.0 | 0.023134323856404317 | 0.023134323856404317 |
| analyst | 0.0 | 0.018634139867648745 | 0.018634139867648745 |
| customer service | 0.0 | 0.018362681047951693 | 0.018362681047951693 |
| project management | 0.0 | 0.018127416737547582 | 0.018127416737547582 |
| research | 0.0 | 0.014652743845425316 | 0.014652743845425316 |
| human resources | 0.0 | 0.012294068323168709 | 0.012294068323168709 |

## Largest Negative Skill-Share Deltas

These are first-vs-last bucket deltas, not evidence that skills are declining over time.

| skill | first_share | last_share | share_delta |
| --- | --- | --- | --- |
| manufacturing | 0.5 | 0.08742180477888171 | -0.41257819522111827 |
| management | 0.5 | 0.09969777584740395 | -0.40030222415259603 |

## Figures

- `reports\figures_100k\job_volume_by_month.png`
- `reports\figures_100k\centroid_drift_by_month.png`
- `reports\figures_100k\top_rising_skills.png`
- `reports\figures_100k\top_declining_skills.png`
- `reports\figures_100k\job_cluster_map_svd.png`
- `reports\figures_100k\similarity_distribution.png`

The figure filenames are historical. Interpret `top_rising_skills.png` and `top_declining_skills.png` as positive and negative skill-share deltas between the two buckets.

## Optional Clustering Summary

The cluster map is exploratory. It is useful for checking whether the representation has structure, but it is not a temporal evaluation metric.

| cluster_id | n_jobs | top_titles |
| --- | --- | --- |
| 0 | 22052 | ['Receptionist', 'Account Executive', 'Sales Representative', 'US Experienced Financial Advisor', 'Customer Service Representative'] |
| 1 | 10994 | ['Registered Nurse', 'Registered Nurse - RN - LTAC', 'Patient Care Technician', 'Medical Assistant', 'Nurse Practitioner'] |
| 2 | 5012 | ['ASSISTANT STORE MANAGER', 'Store Manager', 'Assistant Store Manager', 'CUSTOMER SERVICE REPRESENTATIVE', 'OPERATIONS ASSISTANT MANAGER'] |
| 3 | 22296 | ['Administrative Assistant', 'Senior Accountant', 'Package Handler - Part Time (Warehouse like)', 'Sales Executive', 'Staff Accountant'] |
| 4 | 22970 | ['Project Manager', 'Senior Accountant', 'Controller', 'Staff Accountant', 'Account Manager'] |
| 5 | 15237 | ['Maintenance Technician', 'Salesperson', 'Service Technician', 'Store Driver', 'Delivery Specialist'] |
| 6 | 519 | ['Sales Manager', 'Business Development Manager', 'VP of Sales'] |
| 7 | 920 | ['CERTIFIED NURSING ASSISTANT - OAK FOREST HEALTH & REHAB CENTER', 'LICENSED PRACTICAL NURSE - BERMUDA COMMONS', 'CERTIFIED NURSING ASSISTANT - MARY GRAN', 'LICENSED PRACTICAL NURSE - PINEHURST HEALTHCARE & REHABILITATION CENTER', 'LICENSED PRACTICAL NURSE - THE PAVILION HEALTH CENTER'] |

## Optional Similarity Summary

- Jobs sampled: 500
- Pair similarities: 249,500
- Mean similarity: 0.5409
- Median similarity: 0.5465

## 10k vs 100k Comparison

- 10k selected rows: 10,000 across 2024-03 to 2024-04.
- 100k selected rows: 100,000 across 2024-03 to 2024-04.
- March-to-April centroid distance changed from 0.1914 at 10k to 0.1701 at 100k.
- This shows sensitivity of the demo estimate to sample size. It does not validate a stable temporal trend because both runs still compare the same two buckets with low March support.

## Known Limitations

- TF-IDF/SVD is a fast baseline, not a semantic embedding model.
- Numeric `listed_time` values are interpreted as epoch milliseconds or seconds when plain pandas parsing lands before 1990.
- Skill-share comparison depends on the quality and consistency of `skills_text`.
- Month-to-month drift is noisy when a compared month has few postings.
- This dataset has limited temporal coverage and should not be used to claim stable evolution.

## Next Step

Use `temporal-audit` to make coverage warnings visible before running larger demos. Keep this run as a TF-IDF/SVD temporal drift baseline; use the mock semantic embedding path for schema-safe smoke tests before any explicit small Qwen3 run. FAISS is out of scope until the repo needs approximate nearest-neighbor retrieval at scale.
