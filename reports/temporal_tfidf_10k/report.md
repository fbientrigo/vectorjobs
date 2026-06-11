# M2.3 Temporal Drift Demo

## Command
`python -m jobsrec.cli temporal-demo --input data\silver\jobs.parquet --output-dir reports\temporal_tfidf_10k --figures-dir reports\temporal_tfidf_10k\figures --report-path reports\temporal_tfidf_10k\report.md --sample-size 10000 --sampling-mode temporal-stride --representation tfidf_svd --embedding-backend mock --embedding-model deterministic-mock --embedding-batch-size 16 --embedding-cache-dir data\cache\embeddings --device cpu --max-embedding-rows 1000 --time-column listed_time --centroid-weighting unweighted`

## Run Summary
- Input path: `data\silver\jobs.parquet`
- Rows input: 123849
- Rows selected: 10000
- listed_time parse success rate: 1.0000
- Time column: `listed_time` (canonical posting-listing time)
- Sampling mode: temporal-stride
- Representation: `tfidf_svd`
- Centroid weighting: `unweighted`
- Embedding backend: `None`
- Embedding model: `None`
- Month range: 2024-03 to 2024-04
- Months covered: 2
- Reliability label: `limited_temporal_coverage`
- Runtime seconds: 58.10

## Reliability Gates
- Only 2 months are available; this is a temporal comparison, not a stable trend.
- Only 2 months are available; this is a two-bucket comparison, not a trend.
- Months below 1000 rows may make first/last or consecutive comparisons noisy: 2024-03=1
- First/last comparison is likely noisy because an endpoint month has low support.

## Monthly Row Counts
| month | rows |
| --- | --- |
| 2024-03 | 1 |
| 2024-04 | 9999 |

## Top 10 Largest Centroid Drifts
| month | previous_month | centroid_drift | jobs_in_month |
| --- | --- | --- | --- |
| 2024-04 | 2024-03 | 0.19375077079013114 | 9999 |

## Top 15 Rising Skills
| skill | first_share | last_share | share_delta |
| --- | --- | --- | --- |
| information technology | 0.0 | 0.12382671480144404 | 0.12382671480144404 |
| sales | 0.0 | 0.10054151624548736 | 0.10054151624548736 |
| health care provider | 0.0 | 0.08038507821901324 | 0.08038507821901324 |
| business development | 0.0 | 0.0638387484957882 | 0.0638387484957882 |
| engineering | 0.0 | 0.060770156438026475 | 0.060770156438026475 |
| other | 0.0 | 0.0592057761732852 | 0.0592057761732852 |
| finance | 0.0 | 0.03868832731648616 | 0.03868832731648616 |
| marketing | 0.0 | 0.025992779783393503 | 0.025992779783393503 |
| accounting/auditing | 0.0 | 0.023826714801444042 | 0.023826714801444042 |
| administrative | 0.0 | 0.02328519855595668 | 0.02328519855595668 |
| customer service | 0.0 | 0.020637785800240675 | 0.020637785800240675 |
| analyst | 0.0 | 0.0184115523465704 | 0.0184115523465704 |
| project management | 0.0 | 0.017027677496991578 | 0.017027677496991578 |
| research | 0.0 | 0.014500601684717208 | 0.014500601684717208 |
| human resources | 0.0 | 0.01197352587244284 | 0.01197352587244284 |

## Top 15 Declining Skills
| skill | first_share | last_share | share_delta |
| --- | --- | --- | --- |
| manufacturing | 0.5 | 0.09019253910950661 | -0.40980746089049336 |
| management | 0.5 | 0.10264741275571601 | -0.397352587244284 |

## Figures
- `reports\temporal_tfidf_10k\figures\job_volume_by_month.png`
- `reports\temporal_tfidf_10k\figures\centroid_drift_by_month.png`
- `reports\temporal_tfidf_10k\figures\top_rising_skills.png`
- `reports\temporal_tfidf_10k\figures\top_declining_skills.png`
- `reports\temporal_tfidf_10k\figures\job_cluster_map_svd.png`
- `reports\temporal_tfidf_10k\figures\similarity_distribution.png`

## Optional Clustering Summary
| cluster_id | n_jobs | top_titles |
| --- | --- | --- |
| 0 | 1488 | ['ASSISTANT STORE MANAGER', 'Sales Associate', 'Store Manager', 'Maintenance Technician', 'Assistant Store Manager'] |
| 1 | 1073 | ['Registered Nurse', 'Registered Nurse - RN - LTAC', 'Medical Assistant', 'Certified Nursing Assistant (CNA)', 'Nurse Practitioner'] |
| 2 | 52 | ['Sales Manager'] |
| 3 | 82 | ['REGISTERED NURSE - MARY GRAN', 'CERTIFIED NURSING ASSISTANT - OAK FOREST HEALTH & REHAB CENTER', 'CERTIFIED NURSING ASSISTANT (CNA) - SOUTHPORT HEALTH AND REHABILITATION CENTER', 'PHYSICAL THERAPY ASSISTANT (PTA) - PARKVIEW HEALTH & REHABILITATION CENTER', 'LICENSED PRACTICAL NURSE - WOODLANDS NURSING AND REHABILITATION CENTER'] |
| 4 | 2206 | ['Project Manager', 'Controller', 'Sales Specialist', 'Human Resources Generalist', 'Program Manager'] |
| 5 | 2265 | ['Retail Sales Associate', 'Receptionist', 'Sales Consultant', 'Sales Representative', 'Cashier'] |
| 6 | 970 | ['Senior Software Engineer', 'Software Engineer', 'Full Stack Engineer', 'Network Engineer', 'Electrical Engineer'] |
| 7 | 1864 | ['Customer Service Representative', 'Administrative Assistant', 'Account Manager', 'Mortgage Loan Officer', 'Licensed Therapist for Online Counseling'] |

## Optional Similarity Summary
- Jobs sampled: 500
- Pair similarities: 249500
- Mean similarity: 0.5490
- Median similarity: 0.5800

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
