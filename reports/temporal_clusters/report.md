# Temporal Cluster Analytics

## Dataset
- Date range: 2024-04-05T19:34:32 to 2024-04-20T00:26:56
- Rows input: 123849
- Rows used: 10000
- Time bin: `D`
- Fixed clusters: 12

## Detected Schema
- Text columns: job_card_text, description, title
- Time column: `listed_time`
- Skills column: `skills_text`
- Salary available: True
- Decay available: True

## Top Clusters by Volume
| cluster_id | cluster_label | n_jobs |
| --- | --- | --- |
| 3 | C03 \| data / marketing / business | 1179 |
| 1 | C01 \| job / required / company | 1170 |
| 2 | C02 \| project / management / construction | 1157 |
| 7 | C07 \| clients / customer / client | 1038 |
| 0 | C00 \| care / patient / nursing | 1036 |
| 6 | C06 \| care / health / students | 836 |
| 8 | C08 \| sales / business / customer | 818 |
| 11 | C11 \| design / software / engineer | 787 |
| 10 | C10 \| store / sales / customer | 631 |
| 4 | C04 \| maintenance / equipment / electrical | 578 |
| 5 | C05 \| accounting / financial / tax | 399 |
| 9 | C09 \| food / guest / guests | 371 |

## Top Growing Clusters
| cluster_id | cluster_label | first_share | last_share | share_delta |
| --- | --- | --- | --- | --- |
| 3 | C03 \| data / marketing / business | 0.1099476439790576 | 0.37037037037037035 | 0.26042272639131275 |
| 9 | C09 \| food / guest / guests | 0.02617801047120419 | 0.03889118742242449 | 0.012713176951220304 |
| 10 | C10 \| store / sales / customer | 0.05235602094240838 | 0.05833678113363674 | 0.005980760191228361 |
| 2 | C02 \| project / management / construction | 0.14659685863874344 | 0.14814814814814814 | 0.0015512895094046963 |
| 6 | C06 \| care / health / students | 0.07329842931937172 | 0.07407407407407407 | 0.0007756447547023482 |
| 11 | C11 \| design / software / engineer | 0.07329842931937172 | 0.07407407407407407 | 0.0007756447547023482 |
| 5 | C05 \| accounting / financial / tax | 0.04450261780104712 | 0.041787339677285894 | -0.002715278123761225 |
| 7 | C07 \| clients / customer / client | 0.08900523560209424 | 0.07407407407407407 | -0.014931161528020168 |

## Top Declining Clusters
| cluster_id | cluster_label | first_share | last_share | share_delta |
| --- | --- | --- | --- | --- |
| 0 | C00 \| care / patient / nursing | 0.11518324607329843 | 0.07407407407407407 | -0.041109171999224356 |
| 4 | C04 \| maintenance / equipment / electrical | 0.07068062827225131 | 0.037037037037037035 | -0.03364359123521428 |
| 1 | C01 \| job / required / company | 0.10471204188481675 | 0.07407407407407407 | -0.030637967810742683 |
| 8 | C08 \| sales / business / customer | 0.09424083769633508 | 0.07407407407407407 | -0.02016676362226101 |
| 7 | C07 \| clients / customer / client | 0.08900523560209424 | 0.07407407407407407 | -0.014931161528020168 |
| 5 | C05 \| accounting / financial / tax | 0.04450261780104712 | 0.041787339677285894 | -0.002715278123761225 |

## Salary Availability
- Rows with usable salary: 2875
- Salary coverage: 0.2875

## Decay Availability
- Available: True
- Reason/status: Lifecycle duration data detected; exponential right-censored fit was attempted by cluster.

## Primary Plots
- Bubble timeline: `reports\temporal_clusters\cluster_bubble_timeline.png`
- Share timeseries: `reports\temporal_clusters\cluster_share_timeseries.png`
- Semantic trajectory: `reports\temporal_clusters\cluster_semantic_trajectory.png`

The bubble timeline focuses on four interpretable sectors selected from the fixed clusters: health, tech, sales, and construction. The x-axis is daily posting date, the y-axis is posting share, color identifies the sector, and bubble size is posting count.

The share timeseries tracks the largest fixed clusters over time. Labels come from cluster terms and skills, not raw IDs.

The survival plot uses marker-only survival estimates for the focus sectors. The exponential fit plot overlays observed survival points with fitted exponential survival curves for clusters with enough lifecycle events.

The semantic trajectory plot is a 2D projection for interpretation only. Clustering is performed in the embedding space before projection.

## Output Tables
- Metrics: `reports\temporal_clusters\cluster_time_metrics.parquet`
- Labels: `reports\temporal_clusters\cluster_labels.csv`
- Assignments: `reports\temporal_clusters\cluster_assignments.parquet`

## Decay Summary
| cluster_id | n_events | n_censored | lambda | half_life_days |
| --- | --- | --- | --- | --- |
| 0 | 42 | 0 | 20.983849236409476 | 0.033032413297997415 |
| 1 | 4 | 0 | nan | nan |
| 2 | 8 | 0 | 70.20822752666328 | 0.0098727343643123 |
| 3 | 4 | 0 | nan | nan |
| 4 | 3 | 0 | nan | nan |
| 5 | 2 | 0 | nan | nan |
| 6 | 8 | 0 | 39.97224149895906 | 0.017340713319216685 |
| 7 | 2 | 0 | nan | nan |
| 8 | 1 | 0 | nan | nan |
| 10 | 2 | 0 | nan | nan |
| 11 | 9 | 0 | 16.045561470843136 | 0.043198686553878686 |

## Limitations
- TF-IDF + SVD is a CPU-safe semantic baseline, not a deep language model.
- Clusters are fixed for the selected corpus; changing max rows or random seed can change the fitted global clusters.
- Salary metrics only use rows with usable annualized salary and USD/blank currency.
- Temporal bins with few postings can make cluster shares and drift noisy.
