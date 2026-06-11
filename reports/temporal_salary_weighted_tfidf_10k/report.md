# M2.3 Temporal Drift Demo

## Command
`python -m jobsrec.cli temporal-demo --input data\silver\jobs.parquet --output-dir reports\temporal_salary_weighted_tfidf_10k --figures-dir reports\temporal_salary_weighted_tfidf_10k\figures --report-path reports\temporal_salary_weighted_tfidf_10k\report.md --sample-size 10000 --sampling-mode temporal-stride --representation tfidf_svd --embedding-backend mock --embedding-model deterministic-mock --embedding-batch-size 16 --embedding-cache-dir data\cache\embeddings --device cpu --max-embedding-rows 1000 --time-column original_listed_time --centroid-weighting both`

## Run Summary
- Input path: `data\silver\jobs.parquet`
- Rows input: 123849
- Rows selected: 10000
- listed_time parse success rate: 1.0000
- Time column: `original_listed_time` (wider but sparse original listing time)
- Sampling mode: temporal-stride
- Representation: `tfidf_svd`
- Centroid weighting: `both`
- Embedding backend: `None`
- Embedding model: `None`
- Month range: 2023-12 to 2024-04
- Months covered: 5
- Reliability label: `limited_temporal_coverage`
- Runtime seconds: 74.16

## Reliability Gates
- Only 5 months are available; this is a temporal comparison, not a stable trend.
- Months below 1000 rows may make first/last or consecutive comparisons noisy: 2023-12=2, 2024-01=1, 2024-02=2, 2024-03=152
- First/last comparison is likely noisy because an endpoint month has low support.

## Monthly Row Counts
| month | rows |
| --- | --- |
| 2023-12 | 2 |
| 2024-01 | 1 |
| 2024-02 | 2 |
| 2024-03 | 152 |
| 2024-04 | 9843 |

## Top 10 Largest Centroid Drifts
| month | previous_month | centroid_drift | jobs_in_month |
| --- | --- | --- | --- |
| 2024-02 | 2024-01 | 0.42857467019560647 | 2 |
| 2024-01 | 2023-12 | 0.33030579745143407 | 1 |
| 2024-03 | 2024-02 | 0.225794529534741 | 152 |
| 2024-04 | 2024-03 | 0.01092554502273313 | 9843 |

## Top 15 Rising Skills
| skill | first_share | last_share | share_delta |
| --- | --- | --- | --- |
| information technology | 0.0 | 0.11857055589492975 | 0.11857055589492975 |
| manufacturing | 0.0 | 0.08808796579108125 | 0.08808796579108125 |
| business development | 0.0 | 0.06572999389126451 | 0.06572999389126451 |
| engineering | 0.0 | 0.06108735491753207 | 0.06108735491753207 |
| other | 0.0 | 0.05956017104459377 | 0.05956017104459377 |
| finance | 0.0 | 0.03958460598656078 | 0.03958460598656078 |
| marketing | 0.0 | 0.028222357971899818 | 0.028222357971899818 |
| accounting/auditing | 0.0 | 0.024923640806353085 | 0.024923640806353085 |
| administrative | 0.0 | 0.023213194868662187 | 0.023213194868662187 |
| customer service | 0.0 | 0.01997556505803299 | 0.01997556505803299 |
| project management | 0.0 | 0.01795968234575443 | 0.01795968234575443 |
| analyst | 0.0 | 0.016860109957238852 | 0.016860109957238852 |
| human resources | 0.0 | 0.013439218081857055 | 0.013439218081857055 |
| research | 0.0 | 0.01209529627367135 | 0.01209529627367135 |
| legal | 0.0 | 0.01209529627367135 | 0.01209529627367135 |

## Top 15 Declining Skills
| skill | first_share | last_share | share_delta |
| --- | --- | --- | --- |
| health care provider | 0.3333333333333333 | 0.08051313378130727 | -0.25282019955202606 |
| management | 0.3333333333333333 | 0.09926695174098961 | -0.23406638159234372 |
| sales | 0.3333333333333333 | 0.1034208918753818 | -0.22991244145795153 |

## Figures
- `reports\temporal_salary_weighted_tfidf_10k\figures\centroid_drift_salary_weighted_by_month.png`
- `reports\temporal_salary_weighted_tfidf_10k\figures\salary_coverage_by_month.png`
- `reports\temporal_salary_weighted_tfidf_10k\figures\job_volume_by_month.png`
- `reports\temporal_salary_weighted_tfidf_10k\figures\centroid_drift_by_month.png`
- `reports\temporal_salary_weighted_tfidf_10k\figures\top_rising_skills.png`
- `reports\temporal_salary_weighted_tfidf_10k\figures\top_declining_skills.png`
- `reports\temporal_salary_weighted_tfidf_10k\figures\job_cluster_map_svd.png`
- `reports\temporal_salary_weighted_tfidf_10k\figures\similarity_distribution.png`

## Optional Clustering Summary
| cluster_id | n_jobs | top_titles |
| --- | --- | --- |
| 0 | 2115 | ['Customer Service Representative', 'Assistant Store Manager', 'Receptionist', 'Executive Assistant', 'Team Member'] |
| 1 | 208 | ['Registered Nurse - RN - LTAC', 'Registered Nurse - RN - Rehab', 'Med-Surg Registered Nurse', 'Emergency Room Registered Nurse', 'Med-Surg/Telemetry Registered Nurse'] |
| 2 | 2690 | ['Project Manager', 'Executive Assistant', 'Business Development Manager', 'Sales Specialist', 'Senior Accountant'] |
| 3 | 50 | ['Sales Manager'] |
| 4 | 1531 | ['Maintenance Technician', 'ASSISTANT STORE MANAGER', 'Sales Associate', 'CUSTOMER SERVICE REPRESENTATIVE', 'Salesperson'] |
| 5 | 1007 | ['Registered Nurse', 'Medical Assistant', 'Patient Care Technician', 'Registered Nurse (RN)', 'Certified Nursing Assistant (CNA)'] |
| 6 | 83 | ['CERTIFIED NURSING ASSISTANT - OAK FOREST HEALTH & REHAB CENTER', 'CERTIFIED NURSING ASSISTANT - SOUTHWOOD', 'LICENSED PRACTICAL NURSE - PARKVIEW HEALTH AND REHABILITATION CENTER', 'CERTIFIED NURSING ASSISTANT - MARY GRAN', 'LICENSED PRACTICAL NURSE - LIBERTY COMMONS OF COLUMBUS COUNTY'] |
| 7 | 2316 | ['Administrative Assistant', 'Sales Executive', 'Data Analyst', 'Salesperson', 'Senior Accountant'] |

## Optional Similarity Summary
- Jobs sampled: 500
- Pair similarities: 249500
- Mean similarity: 0.5736
- Median similarity: 0.5962

## Salary-Weighted Centroid View
- Salary-weighted centroids describe the salary-disclosed USD subset, not the full job market.
- Salary rows used: 2852
- Salary coverage in selected rows: 0.2852
- Salary weighting strategy: `normalized_salary_else_annualized_median_else_annualized_midpoint_log1p_month_mean_normalized_clipped`
- Salary-weighted drift path: `reports\temporal_salary_weighted_tfidf_10k\monthly_centroid_drift_salary_weighted.parquet`

## Known Limitations
- This is a temporal comparison / temporal drift demo, not evidence of market evolution unless coverage is sufficient.
- TF-IDF/SVD is a fast baseline, not a semantic embedding model.
- Semantic embedding runs should stay small on 4 GB VRAM / 8 GB RAM machines unless explicitly validated.
- Numeric `listed_time` values are interpreted as epoch milliseconds or seconds when plain pandas parsing lands before 1990.
- Skill growth depends on the quality and consistency of `skills_text`.
- Month-to-month drift can be noisy when a month has few postings.
- FAISS is not needed for this milestone because the demo computes aggregate centroids, not large-scale ANN retrieval.

## Deferred Tasks
- Bootstrap stability for drift and skill growth is deferred; add a small explicit seed list before using confidence intervals.

## Next Step
Run temporal audit first, then use larger samples only if local memory allows.
