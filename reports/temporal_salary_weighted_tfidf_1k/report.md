# M2.3 Temporal Drift Demo

## Command
`python -m jobsrec.cli temporal-demo --input data\silver\jobs.parquet --output-dir reports\temporal_salary_weighted_tfidf_1k --figures-dir reports\temporal_salary_weighted_tfidf_1k\figures --report-path reports\temporal_salary_weighted_tfidf_1k\report.md --sample-size 1000 --sampling-mode temporal-stride --representation tfidf_svd --embedding-backend mock --embedding-model deterministic-mock --embedding-batch-size 16 --embedding-cache-dir data\cache\embeddings --device cpu --max-embedding-rows 1000 --time-column original_listed_time --centroid-weighting both`

## Run Summary
- Input path: `data\silver\jobs.parquet`
- Rows input: 123849
- Rows selected: 1000
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
- Runtime seconds: 15.57

## Reliability Gates
- Only 5 months are available; this is a temporal comparison, not a stable trend.
- Months below 1000 rows may make first/last or consecutive comparisons noisy: 2023-12=2, 2024-01=1, 2024-02=1, 2024-03=15, 2024-04=981
- First/last comparison is likely noisy because an endpoint month has low support.

## Monthly Row Counts
| month | rows |
| --- | --- |
| 2023-12 | 2 |
| 2024-01 | 1 |
| 2024-02 | 1 |
| 2024-03 | 15 |
| 2024-04 | 981 |

## Top 10 Largest Centroid Drifts
| month | previous_month | centroid_drift | jobs_in_month |
| --- | --- | --- | --- |
| 2024-02 | 2024-01 | 0.7648537849629904 | 1 |
| 2024-03 | 2024-02 | 0.5955461132052315 | 15 |
| 2024-01 | 2023-12 | 0.4219126391049215 | 1 |
| 2024-04 | 2024-03 | 0.05478780573563391 | 981 |

## Top 15 Rising Skills
| skill | first_share | last_share | share_delta |
| --- | --- | --- | --- |
| information technology | 0.0 | 0.1363914373088685 | 0.1363914373088685 |
| manufacturing | 0.0 | 0.0856269113149847 | 0.0856269113149847 |
| engineering | 0.0 | 0.07155963302752294 | 0.07155963302752294 |
| business development | 0.0 | 0.062385321100917435 | 0.062385321100917435 |
| other | 0.0 | 0.05688073394495413 | 0.05688073394495413 |
| finance | 0.0 | 0.03669724770642202 | 0.03669724770642202 |
| administrative | 0.0 | 0.023853211009174313 | 0.023853211009174313 |
| accounting/auditing | 0.0 | 0.021406727828746176 | 0.021406727828746176 |
| customer service | 0.0 | 0.02018348623853211 | 0.02018348623853211 |
| marketing | 0.0 | 0.01834862385321101 | 0.01834862385321101 |
| project management | 0.0 | 0.01834862385321101 | 0.01834862385321101 |
| research | 0.0 | 0.01712538226299694 | 0.01712538226299694 |
| analyst | 0.0 | 0.01712538226299694 | 0.01712538226299694 |
| human resources | 0.0 | 0.014067278287461774 | 0.014067278287461774 |
| education | 0.0 | 0.01345565749235474 | 0.01345565749235474 |

## Top 15 Declining Skills
| skill | first_share | last_share | share_delta |
| --- | --- | --- | --- |
| health care provider | 0.3333333333333333 | 0.07217125382262997 | -0.26116207951070336 |
| sales | 0.3333333333333333 | 0.09724770642201835 | -0.23608562691131496 |
| management | 0.3333333333333333 | 0.10091743119266056 | -0.23241590214067276 |

## Figures
- `reports\temporal_salary_weighted_tfidf_1k\figures\job_volume_by_month.png`
- `reports\temporal_salary_weighted_tfidf_1k\figures\centroid_drift_by_month.png`
- `reports\temporal_salary_weighted_tfidf_1k\figures\top_rising_skills.png`
- `reports\temporal_salary_weighted_tfidf_1k\figures\top_declining_skills.png`
- `reports\temporal_salary_weighted_tfidf_1k\figures\job_cluster_map_svd.png`
- `reports\temporal_salary_weighted_tfidf_1k\figures\similarity_distribution.png`

## Optional Clustering Summary
| cluster_id | n_jobs | top_titles |
| --- | --- | --- |
| 0 | 108 | ['Maintenance Technician', 'Commercial Appraiser Manager', 'Supervisor Building Operations', 'Food Service Specialist', 'Adena Lakeland - Ironworker'] |
| 1 | 243 | ['Software Engineer', 'Post Office Assistant - Term - St-Denis-de-Brompton / Kingsburry', 'Retail Sales Associate', 'Shift Leader', 'Scrum Master (Coach)'] |
| 2 | 143 | ['Project Manager', 'Engineering Manager', 'Network Engineer', 'Senior Power Platform Developer-Robotic Process Automation (RPA) Developer', 'Data Scientist - Clearance Required with Security Clearance'] |
| 3 | 182 | ['Executive Assistant', 'Group Systems Accountant', 'Field Office ISSM - Open Rank-RS-Albuquerque, NM', 'Part time Recruiter', 'Regional Sales Director'] |
| 4 | 109 | ['Registered Nurse', 'Medical Receptionist', 'Registered Nurse RN - Medical ICU MICU - PT Nights', 'Licensed Practical Nurse', 'Behavior Technician - ST'] |
| 5 | 6 | ['salesperson', 'Salesperson', 'Retail Parts Pro', 'Store Driver'] |
| 6 | 4 | ['Nurse - RN - LTAC', 'Nurse - RN - Med Surg', 'Physical Therapist / PT - Rehab', 'Nurse - RN - Labor and Delivery / L&D'] |
| 7 | 205 | ['Sales Manager', 'Customer Service Representative', 'Business Development Manager', 'Software Engineer', 'Associate Attorney'] |

## Optional Similarity Summary
- Jobs sampled: 500
- Pair similarities: 249500
- Mean similarity: 0.4916
- Median similarity: 0.5055

## Salary-Weighted Centroid View
- Salary-weighted centroids describe the salary-disclosed USD subset, not the full job market.
- Salary rows used: 276
- Salary coverage in selected rows: 0.2760
- Salary weighting strategy: `normalized_salary_else_annualized_median_else_annualized_midpoint_log1p_month_mean_normalized_clipped`
- Salary-weighted drift path: `reports\temporal_salary_weighted_tfidf_1k\monthly_centroid_drift_salary_weighted.parquet`

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
