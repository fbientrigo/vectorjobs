# Graph Report - .  (2026-06-13)

## Corpus Check
- Corpus is ~28,677 words - fits in a single context window. You may not need a graph.

## Summary
- 585 nodes · 978 edges · 41 communities (27 shown, 14 thin omitted)
- Extraction: 76% EXTRACTED · 24% INFERRED · 0% AMBIGUOUS · INFERRED: 234 edges (avg confidence: 0.74)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Silver Data Profiling|Silver Data Profiling]]
- [[_COMMUNITY_Top-k Retrieval Engine|Top-k Retrieval Engine]]
- [[_COMMUNITY_Temporal Clustering Analytics|Temporal Clustering Analytics]]
- [[_COMMUNITY_Silver Dataset Building|Silver Dataset Building]]
- [[_COMMUNITY_Schema and Data Contracts|Schema and Data Contracts]]
- [[_COMMUNITY_Metadata Preservation Verification|Metadata Preservation Verification]]
- [[_COMMUNITY_Job Card Text Construction|Job Card Text Construction]]
- [[_COMMUNITY_Skill Share Evolution Analysis|Skill Share Evolution Analysis]]
- [[_COMMUNITY_Temporal Clustering Analytics|Temporal Clustering Analytics]]
- [[_COMMUNITY_Temporal Trends & Centroid Drift|Temporal Trends & Centroid Drift]]
- [[_COMMUNITY_Top-k Retrieval Engine|Top-k Retrieval Engine]]
- [[_COMMUNITY_TF-IDF Vectorization Wrapper|TF-IDF Vectorization Wrapper]]
- [[_COMMUNITY_Schema and Data Contracts|Schema and Data Contracts]]
- [[_COMMUNITY_Temporal Trends & Centroid Drift|Temporal Trends & Centroid Drift]]
- [[_COMMUNITY_Temporal Trends & Centroid Drift|Temporal Trends & Centroid Drift]]
- [[_COMMUNITY_Temporal Trends & Centroid Drift|Temporal Trends & Centroid Drift]]
- [[_COMMUNITY_Schema and Data Contracts|Schema and Data Contracts]]
- [[_COMMUNITY_Mock Embedding Backend|Mock Embedding Backend]]
- [[_COMMUNITY_Qwen3 Dense Embedding Backend|Qwen3 Dense Embedding Backend]]
- [[_COMMUNITY_Temporal Trends & Centroid Drift|Temporal Trends & Centroid Drift]]
- [[_COMMUNITY_Schema and Data Contracts|Schema and Data Contracts]]
- [[_COMMUNITY_Temporal Trends & Centroid Drift|Temporal Trends & Centroid Drift]]
- [[_COMMUNITY_Group jobsrec – job-skill recomme...|Group: jobsrec – job-skill recomme...]]
- [[_COMMUNITY_Group __init__.py|Group: __init__.py]]
- [[_COMMUNITY_Embedding Backend & Storage|Embedding Backend & Storage]]
- [[_COMMUNITY_Embedding Backend & Storage|Embedding Backend & Storage]]
- [[_COMMUNITY_Embedding Backend & Storage|Embedding Backend & Storage]]
- [[_COMMUNITY_Embedding Backend & Storage|Embedding Backend & Storage]]
- [[_COMMUNITY_TF-IDF Vectorization Wrapper|TF-IDF Vectorization Wrapper]]
- [[_COMMUNITY_TF-IDF Vectorization Wrapper|TF-IDF Vectorization Wrapper]]
- [[_COMMUNITY_TF-IDF Vectorization Wrapper|TF-IDF Vectorization Wrapper]]
- [[_COMMUNITY_Top-k Retrieval Engine|Top-k Retrieval Engine]]
- [[_COMMUNITY_Top-k Retrieval Engine|Top-k Retrieval Engine]]
- [[_COMMUNITY_Group Milestone Roadmap|Group: Milestone Roadmap]]

## God Nodes (most connected - your core abstractions)
1. `profile_silver()` - 36 edges
2. `build_silver()` - 28 edges
3. `run_temporal_clusters()` - 27 edges
4. `TfidfBackend` - 22 edges
5. `_silver_df()` - 22 edges
6. `_make_silver_df()` - 21 edges
7. `TfidfRetriever` - 20 edges
8. `build_job_card_text()` - 20 edges
9. `run_skill_evolution()` - 19 edges
10. `run_temporal_demo()` - 19 edges

## Surprising Connections (you probably didn't know these)
- `build_embeddings_cmd()` --calls--> `FakeDenseEmbeddingBackend`  [INFERRED]
  src/jobsrec/cli.py → tests/test_qwen3_backend_contract.py
- `_silver_df()` --calls--> `build_silver()`  [INFERRED]
  tests/test_m1_5_metadata_preservation.py → src/jobsrec/data/load.py
- `silver_path()` --calls--> `build_silver()`  [INFERRED]
  tests/test_m1_profile_and_smoke.py → src/jobsrec/data/load.py
- `first_job_id()` --calls--> `build_silver()`  [INFERRED]
  tests/test_m1_profile_and_smoke.py → src/jobsrec/data/load.py
- `TestOptionalColumnPreservation` --uses--> `SilverProfile`  [INFERRED]
  tests/test_m1_5_metadata_preservation.py → src/jobsrec/data/profile.py

## Hyperedges (group relationships)
- **Pipeline Execution Profiles** — readme_jobsrec_pipeline, configs_colab_t4, configs_local_6gb [INFERRED 0.90]
- **Pipeline Raw Input Schemas** — specs_data_contract_postings, specs_data_contract_salaries, specs_data_contract_silver [INFERRED 0.90]
- **Temporal Trends Prototype Workflow** — specs_temporal_trends_audit, specs_temporal_trends_salary_weight, specs_temporal_trends_mock [INFERRED 0.90]

## Communities (41 total, 14 thin omitted)

### Community 0 - "Silver Data Profiling"
Cohesion: 0.05
Nodes (33): _compute_listed_time_parse_rate(), _compute_skill_stats(), _compute_ts_parse_rate(), profile_silver(), profile_silver_from_path(), Data profiling for silver Parquet datasets.  Produces a structured summary of a, Return a JSON-serialisable dict representation., Compute a data profile for a silver Parquet DataFrame.      Parameters     ----- (+25 more)

### Community 1 - "Top-k Retrieval Engine"
Cohesion: 0.05
Nodes (30): Thin wrapper around ``TfidfVectorizer`` with persistence helpers.      Parameter, TfidfBackend, DenseRetriever, Top-k retrieval over dense embeddings., Cosine-similarity retriever backed by dense numpy embeddings.      Assumes embed, Return the top-k most similar jobs to *query_job_id*.          The query job its, Top-k cosine-similarity retrieval over a sparse TF-IDF matrix.  Design constrain, Return the matrix row index for *job_id*, or raise ``KeyError``. (+22 more)

### Community 2 - "Temporal Clustering Analytics"
Cohesion: 0.07
Nodes (58): _cluster_jobs(), test_cluster_labels_are_non_empty(), test_fixed_cluster_assignment_is_stable_with_random_state(), test_no_closure_or_last_seen_columns_skip_survival(), test_survival_exponential_fit_returns_expected_lambda(), test_temporal_aggregation_produces_expected_bins_shares_and_salary_coverage(), test_temporal_clusters_cli_writes_required_outputs_and_skip_artifact(), build_cluster_labels() (+50 more)

### Community 3 - "Silver Dataset Building"
Cohesion: 0.06
Nodes (33): aggregate_salaries(), build_silver(), build_skills_text(), load_job_skills(), load_postings(), load_salaries(), load_skills_mapping(), _make_manifest() (+25 more)

### Community 4 - "Schema and Data Contracts"
Cohesion: 0.08
Nodes (14): assert_columns(), Column-level schema contracts for every CSV / Parquet consumed by jobsrec.  Vali, Check that *df* contains every column listed in *required*.      Parameters, Like :func:`validate_columns` but raises ``ValueError`` on failure.      Paramet, Outcome of a schema validation check., validate_columns(), ValidationResult, Tests for the data contract schema module.  Covers: * validate_columns returns c (+6 more)

### Community 5 - "Metadata Preservation Verification"
Cohesion: 0.09
Nodes (13): Milestone 1.5 — tests for temporal/salary/metadata preservation in silver.  Cove, Job 1010 has 'not_a_timestamp' — it should be preserved (NaN or raw)., Salary columns survive into silver without duplicating rows., Jobs 1003 and 1009 have no salaries.csv row — must be null., Jobs with no job_skills rows survive build_silver., All 10 kaggle_minimal jobs must be in silver., If a job_id has no skills, skills_text should be empty string., Build silver from kaggle_minimal and return the DataFrame. (+5 more)

### Community 6 - "Job Card Text Construction"
Cohesion: 0.1
Nodes (13): Tests for the job_card_text builder.  Covers: * Determinism — calling build_job_, Calling with only required args should not raise., Return the section label tokens from each line of *text*., Multiple repeated calls must always match the first., With only title + skills + description, they must still be in order., _sections(), TestContentFidelity, TestDeterminism (+5 more)

### Community 7 - "Skill Share Evolution Analysis"
Cohesion: 0.12
Nodes (25): _mock_jobs(), test_normalize_skill(), test_run_skill_evolution_succeeds(), test_skill_evolution_cli(), test_split_skills(), assign_job_domains(), build_skill_long_table(), compute_domain_skill_monthly() (+17 more)

### Community 8 - "Temporal Clustering Analytics"
Cohesion: 0.11
Nodes (25): build_embeddings_cmd(), build_silver_cmd(), build_tfidf_cmd(), _load_config(), main(), profile_silver_cmd(), Click CLI entry-points for the jobsrec pipeline.  Commands -------- build-silver, Fit TF-IDF on silver data and write gold artefacts. (+17 more)

### Community 9 - "Temporal Trends & Centroid Drift"
Cohesion: 0.15
Nodes (24): _build_embedding_backend(), _comparison_note(), compute_centroid_drift(), compute_semantic_centroid_drift(), compute_skill_growth(), _embedding_cache_path(), _format_table(), _load_or_compute_embeddings() (+16 more)

### Community 10 - "Top-k Retrieval Engine"
Cohesion: 0.1
Nodes (13): ABC, EmbeddingBackend, Abstract base class for embedding backends., Protocol for dense embedding backends., build_and_save_dense(), DenseArtifacts, Dense embedding storage and artifact builder., Paths produced by dense embedding builders. (+5 more)

### Community 11 - "TF-IDF Vectorization Wrapper"
Cohesion: 0.11
Nodes (12): fit_and_save(), TF-IDF vectoriser wrapper.  Wraps ``sklearn.feature_extraction.text.TfidfVectori, Transform *documents* with the already-fitted vectoriser.          Parameters, Persist the fitted vectoriser and matrix to *output_dir*.          Creates ``tfi, Fit a :class:`TfidfBackend`, persist all artefacts, write a manifest.      Param, Raise ``TypeError`` if *documents* is not a non-empty list of strings., Paths produced by :func:`fit_and_save`., Fit the vectoriser on *documents* and return the TF-IDF matrix.          Paramet (+4 more)

### Community 12 - "Schema and Data Contracts"
Cohesion: 0.19
Nodes (21): _synthetic_jobs(), test_centroid_drift_output_has_required_columns(), test_compute_temporal_audit_reports_coverage_by_month(), test_invalid_dates_are_excluded_from_temporal_sampling(), test_numeric_epoch_milliseconds_parse_to_real_month(), test_original_listed_time_can_expand_temporal_buckets(), test_random_sampling_is_deterministic(), test_salary_weighted_centroid_drift_outputs_required_columns() (+13 more)

### Community 13 - "Temporal Trends & Centroid Drift"
Cohesion: 0.13
Nodes (13): EmbeddingBackend, MockEmbeddingBackend, Small deterministic backend that does not download models., Qwen3EmbeddingBackend, Qwen3 embedding backend using SentenceTransformers., Initialize the Qwen3 embedding backend.          Parameters         ----------, Encode a list of texts into a dense numpy array.          Parameters         ---, Run a full-dataset temporal audit and write report, JSON, and Parquet outputs. (+5 more)

### Community 14 - "Temporal Trends & Centroid Drift"
Cohesion: 0.22
Nodes (10): test_reliability_gate_labels_two_month_dataset(), test_reliability_gate_sufficient_for_six_supported_months(), test_reliability_gate_warns_when_month_under_1000_rows(), build_reliability_assessment(), compute_temporal_audit(), compute_temporal_column_coverage(), _non_empty_mask(), Classify temporal support and produce reportable warnings. (+2 more)

### Community 15 - "Temporal Trends & Centroid Drift"
Cohesion: 0.25
Nodes (8): test_compute_annual_salary_prefers_normalized_then_annualizes_fallbacks(), test_prepare_salary_weights_filters_non_usd_and_normalizes_by_month(), compute_annual_salary(), compute_salary_weighted_centroid_drift(), prepare_salary_weights(), Return annualized salary values using normalized salary, median, then range midp, Create robust salary weights normalized within month., Compute normalized monthly centroids weighted by robust salary weights.

### Community 16 - "Schema and Data Contracts"
Cohesion: 0.29
Nodes (8): Colab T4 Configuration Profile, Local 6GB CPU Configuration Profile, jobsrec Pipeline Overview, Data Contract Specification, Embedding Backends Specification, Temporal Trends Prototype Specification, Temporal Audit Workflow, Limited Temporal Coverage Constraint

### Community 19 - "Temporal Trends & Centroid Drift"
Cohesion: 0.4
Nodes (5): test_month_parsing_works(), parse_listed_time(), parse_time_column(), Parse listed_time with pandas coercion semantics., Parse a temporal column with pandas coercion and numeric epoch fallback.

### Community 20 - "Schema and Data Contracts"
Cohesion: 0.4
Nodes (5): job_card_text Formatting Rules, Postings Input Schema, Salaries Input Schema & Aggregation, Silver Dataset Schema, Salary-Weighted Centroid Method

### Community 21 - "Temporal Trends & Centroid Drift"
Cohesion: 0.5
Nodes (4): Fake Embedding Backend Mock, EmbeddingBackend Interface, Qwen3 Embedding Backend, Deterministic Semantic Mock Backend

## Knowledge Gaps
- **159 isolated node(s):** `Click CLI entry-points for the jobsrec pipeline.  Commands -------- build-silver`, `jobsrec — LinkedIn job-skill recommendation pipeline.`, `Load raw CSVs, build job_card_text, write silver Parquet.`, `Fit TF-IDF on silver data and write gold artefacts.`, `Return top-k similar jobs as JSON.` (+154 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **14 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `build_silver()` connect `Silver Dataset Building` to `Temporal Clustering Analytics`, `Silver Data Profiling`, `Metadata Preservation Verification`, `Job Card Text Construction`?**
  _High betweenness centrality (0.345) - this node is a cross-community bridge._
- **Why does `build_silver_cmd()` connect `Temporal Clustering Analytics` to `Silver Dataset Building`?**
  _High betweenness centrality (0.209) - this node is a cross-community bridge._
- **Are the 27 inferred relationships involving `profile_silver()` (e.g. with `.test_closed_time_parse_rate_none_when_absent()` and `.test_work_type_distribution_present()`) actually correct?**
  _`profile_silver()` has 27 INFERRED edges - model-reasoned connections that need verification._
- **Are the 18 inferred relationships involving `build_silver()` (e.g. with `build_silver_cmd()` and `build_job_card_text()`) actually correct?**
  _`build_silver()` has 18 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `run_temporal_clusters()` (e.g. with `temporal_clusters_cmd()` and `compute_annual_salary()`) actually correct?**
  _`run_temporal_clusters()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 15 inferred relationships involving `TfidfBackend` (e.g. with `ScoredJob` and `RetrievalResult`) actually correct?**
  _`TfidfBackend` has 15 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Click CLI entry-points for the jobsrec pipeline.  Commands -------- build-silver`, `jobsrec — LinkedIn job-skill recommendation pipeline.`, `Load raw CSVs, build job_card_text, write silver Parquet.` to the rest of the system?**
  _159 weakly-connected nodes found - possible documentation gaps or missing edges._