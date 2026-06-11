# M2.3 Temporal Drift Demo

## Command
`python -m jobsrec.cli temporal-demo --input data\silver\jobs.parquet --output-dir reports\temporal_semantic_mock_1k --figures-dir reports\temporal_semantic_mock_1k\figures --report-path reports\temporal_semantic_mock_1k\report.md --sample-size 1000 --sampling-mode temporal-stride --representation semantic_embeddings --embedding-backend mock --embedding-model deterministic-mock --embedding-batch-size 16 --embedding-cache-dir data\cache\embeddings --device cpu --max-embedding-rows 1000`

## Run Summary
- Input path: `data\silver\jobs.parquet`
- Rows input: 123849
- Rows selected: 1000
- listed_time parse success rate: 1.0000
- Sampling mode: temporal-stride
- Representation: `semantic_embeddings`
- Embedding backend: `mock`
- Embedding model: `deterministic-mock`
- Month range: 2024-03 to 2024-04
- Months covered: 2
- Reliability label: `limited_temporal_coverage`
- Runtime seconds: 21.89

## Reliability Gates
- Only 2 months are available; this is a temporal comparison, not a stable trend.
- Only 2 months are available; this is a two-bucket comparison, not a trend.
- Months below 1000 rows may make first/last or consecutive comparisons noisy: 2024-03=1, 2024-04=999
- First/last comparison is likely noisy because an endpoint month has low support.

## Monthly Row Counts
| month | rows |
| --- | --- |
| 2024-03 | 1 |
| 2024-04 | 999 |

## Top 10 Largest Centroid Drifts
| month | previous_month | centroid_drift | jobs_in_month |
| --- | --- | --- | --- |
| 2024-04 | 2024-03 | 1.0155617510899901 | 999 |

## Top 15 Rising Skills
| skill | first_share | last_share | share_delta |
| --- | --- | --- | --- |
| information technology | 0.0 | 0.12877871825876663 | 0.12877871825876663 |
| sales | 0.0 | 0.10701330108827085 | 0.10701330108827085 |
| health care provider | 0.0 | 0.07859733978234583 | 0.07859733978234583 |
| business development | 0.0 | 0.062273276904474005 | 0.062273276904474005 |
| engineering | 0.0 | 0.06106408706166868 | 0.06106408706166868 |
| other | 0.0 | 0.0592503022974607 | 0.0592503022974607 |
| finance | 0.0 | 0.044740024183796856 | 0.044740024183796856 |
| accounting/auditing | 0.0 | 0.02962515114873035 | 0.02962515114873035 |
| project management | 0.0 | 0.02357920193470375 | 0.02357920193470375 |
| marketing | 0.0 | 0.02357920193470375 | 0.02357920193470375 |
| administrative | 0.0 | 0.018742442563482467 | 0.018742442563482467 |
| customer service | 0.0 | 0.018137847642079808 | 0.018137847642079808 |
| analyst | 0.0 | 0.015114873035066506 | 0.015114873035066506 |
| general business | 0.0 | 0.013905683192261185 | 0.013905683192261185 |
| consulting | 0.0 | 0.012696493349455865 | 0.012696493349455865 |

## Top 15 Declining Skills
| skill | first_share | last_share | share_delta |
| --- | --- | --- | --- |
| manufacturing | 0.5 | 0.08827085852478839 | -0.4117291414752116 |
| management | 0.5 | 0.09915356711003627 | -0.40084643288996374 |

## Figures
- `reports\temporal_semantic_mock_1k\figures\job_volume_by_month.png`
- `reports\temporal_semantic_mock_1k\figures\centroid_drift_by_month.png`
- `reports\temporal_semantic_mock_1k\figures\top_rising_skills.png`
- `reports\temporal_semantic_mock_1k\figures\top_declining_skills.png`
- `reports\temporal_semantic_mock_1k\figures\job_cluster_map_svd.png`
- `reports\temporal_semantic_mock_1k\figures\similarity_distribution.png`

## Optional Clustering Summary
| cluster_id | n_jobs | top_titles |
| --- | --- | --- |
| 0 | 123 | ['Sales Development Representative', 'Physical Therapist', 'Customer Service Representative', 'Retail Sales Associate Apparel', 'IT Buyer / Strategic  Sourcing Manager'] |
| 1 | 112 | ['Food and Beverage Manager', 'Intern: Truck Financial Services Sales Support Analyst (Summer 2024)', 'Inventory Analyst', 'Retail Commission Sales Associate - Fine Jewelry, Biltmore Fashion Park - Full Time', 'Wind Technician I - Allen, NE'] |
| 2 | 109 | ['Account Executive', 'Senior Accountant', 'Director, Finance Business Systems', 'Ultrasound Technician (Cardio Multimodality/Sonographer) *New Hire Signing Bonus*', 'Software Development Engineer, Spark Performance'] |
| 3 | 143 | ['Sr. Packaging Engineer', 'Sr. Team Leader - Advisory & Business Development', 'Service Technician', 'Supportive Counselor I', 'Tax Manager - Financial Services - Wealth and Asset Management Client Delivery Services-EDGE'] |
| 4 | 139 | ['Superintendent', 'H&M Sales Associate - Store Volume', 'Business Analyst', 'Nurse - CNA - LTC', 'Associate Buyer'] |
| 5 | 113 | ['Senior Software Engineer', 'State Tested Nursing Assistant (STNA)', 'Logistics Manager', 'Senior Network Engineer', 'Global Sales Manager'] |
| 6 | 140 | ['Digital Designer', 'Sales Call Center Representative I', 'Pre-School Director/Teacher', 'Manager, Customer Success Engagement *REMOTE*', 'Senior HVAC Technician'] |
| 7 | 121 | ['Salesperson', 'Warehouse Worker/Cover Driver', 'Host', 'Sr. Manager, Public Relations', 'Waterfront Program Director'] |

## Optional Similarity Summary
- Jobs sampled: 500
- Pair similarities: 249500
- Mean similarity: -0.0003
- Median similarity: 0.0000

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
