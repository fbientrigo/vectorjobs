# Skill Evolution (Local TF-IDF, no LLM/Qwen)

## Command
`jobsrec skill-evolution --input data\silver\jobs.parquet --outdir reports\skill_evolution --bin D --top-n 12 --max-rows 10000 --random-state 42 --confidence-threshold 0.05 --margin-threshold 0.01`

## Dataset
- Input: `data\silver\jobs.parquet`
- Date range: 2024-04-05T19:34:32 to 2024-04-20T00:26:56
- Rows input: 123849
- Rows used: 10000
- Time bin: `D`
- Time column: `listed_time`
- Skill column: `skills_text`

## Domain Assignment
Domains are assigned with TF-IDF + SVD job embeddings matched against domain description embeddings (cosine similarity), with a keyword fallback. This replaces the Qwen embedding step from `notebooks/cluster_skill_timelines_colab.ipynb` for local/offline use. No LLM enrichment is used; skills come only from `skills_text`.

| domain | domain_label | n_jobs | n_uncertain |
| --- | --- | --- | --- |
| health | Health | 2357 | 102 |
| tech | Tech | 2228 | 107 |
| construction_industry | Construction/Industry | 1709 | 78 |
| education | Education | 781 | 52 |
| sales | Sales | 2925 | 177 |

## Skill Evolution Plots
- Health: `reports\skill_evolution\skill_evolution_health.png`
- Tech: `reports\skill_evolution\skill_evolution_tech.png`
- Construction/Industry: `reports\skill_evolution\skill_evolution_construction_industry.png`
- Education: `reports\skill_evolution\skill_evolution_education.png`
- Sales: `reports\skill_evolution\skill_evolution_sales.png`

## Top Skills per Domain (overall)
| domain | top_skills |
| --- | --- |
| Health | Health Care Provider (1257); Other (270); Management (230); Manufacturing (196); Information Technology (190); Sales (137); Administrative (88); Business Development (85) |
| Tech | Information Technology (1396); Engineering (745); Analyst (149); Project Management (134); Other (119); Research (112); Management (101); Marketing (96) |
| Construction/Industry | Manufacturing (913); Management (894); Information Technology (195); Engineering (168); Other (166); Project Management (97); Sales (68); Administrative (59) |
| Education | Education (164); Other (147); Training (137); Management (128); Manufacturing (96); Sales (62); Human Resources (59); Administrative (52) |
| Sales | Sales (1450); Business Development (933); Finance (449); Accounting/auditing (336); Management (306); Other (276); Customer Service (246); Marketing (210) |

## Output Tables
- Domain assignments: `reports\skill_evolution\domain_assignments.parquet`
- Skill long table: `reports\skill_evolution\skill_long.parquet`
- Domain skill monthly shares: `reports\skill_evolution\domain_skill_monthly.parquet`

## Reliability
- Label: `sufficient_temporal_coverage`
- Months below 1000 rows may make first/last or consecutive comparisons noisy: 2024-04-05=382, 2024-04-06=667, 2024-04-07=32, 2024-04-09=805, 2024-04-11=544, 2024-04-12=609, 2024-04-15=615, 2024-04-16=367, 2024-04-17=824, 2024-04-20=27
- First/last comparison is likely noisy because an endpoint month has low support.

## Limitations
- TF-IDF + SVD domain assignment is a CPU-safe baseline; it is less accurate than the Qwen-based assignment used in the Colab notebook.
- Skills are deterministic, derived only from the skills column; no LLM enrichment is performed.
- Domain confidence/margin thresholds were tuned for TF-IDF cosine similarities, which run lower than dense sentence-embedding similarities.
- Time bins with few postings can make skill shares noisy.
