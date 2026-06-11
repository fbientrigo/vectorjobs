# Milestone 1.5: Real Dataset Data Contract & Pipeline Validation

This report documents the successful implementation of Milestone 1.5, verifying that the pipeline preserves the rich metadata provided by the real Kaggle dataset and handles complex data joins without duplicating records.

## 1. Goal 

The goal of this milestone was to:
1. Update the `jobsrec` schema and data loaders to preserve real Kaggle metadata columns (temporal, job, salary, geo, application).
2. Introduce a deterministic salary join combining `postings.csv` with `jobs/salaries.csv` to capture richer compensation data without exploding the row count.
3. Keep the pipeline robust enough to handle the missing elements common in real datasets (e.g., jobs with no skills in `job_skills.csv`).
4. Prove it all works against the full ~123k real-world dataset.

## 2. Architecture & Design Decisions

- **Deterministic Salary Join**: `jobs/salaries.csv` contains multiple rows for a single `job_id`. We introduced an `aggregate_salaries` operation to collapse these. Numeric bounds (`min_salary`, `max_salary`, `med_salary`) take the maximum available value. Strings (`pay_period`, `currency`) take the first non-null value. This perfectly maintains a 1:1 join relationship.
- **Handling Overlapping Suffixes**: Both `postings.csv` and `salaries.csv` contain identically named salary columns. We actively drop these overlap columns from `postings.csv` prior to the merge so that the final Silver dataset stays clean without `_x` or `_y` suffix pollution.
- **Optional Column Flow**: We broadened `POSTINGS_OPTIONAL` to include 19 fields (timestamps, work types, remote flags). The `build-silver` script dynamically passes these columns forward directly into the `.parquet` file. 
- **Time Parsing Diagnoses**: Kaggle timestamps arrive as raw Unix millisecond integers. We do not cast them to datetime objects in the data contract (keeping it simple and native), but the `profile-silver` command calculates the parse success rates using `pd.to_datetime(unit="ms")` to prove the timestamps are valid representations.

## 3. Results on Real Dataset

We ran the complete `build-silver`, `profile-silver`, `build-tfidf`, and `recommend` pipeline on a 5k subset of the real data, and then ran the heavy build/profile operations on the full 123k row Kaggle drop.

### 3.1 Data Profile (Full Dataset)
- **Total Postings Processed**: `123,849`
- **Output Rows**: `123,849` (Confirming zero duplication from the salary join!)
- **Jobs without skills**: `1,753` (Successfully preserved rather than dropped)
- **Timestamp Integrity**: `listed_time` and `expiry` show a 100% parse success rate. 

### 3.2 Salary Coverage
Out of the 123,849 postings, the join successfully appended structural salary data:
- `normalized_salary`: 123,849 (100% coverage, directly from postings)
- `min_salary` / `max_salary`: 29,793
- `med_salary`: 6,280
- `pay_period` / `currency`: 36,073

### 3.3 Metadata Distributions

*Work Types:*
- Full-time: 98,814
- Contract: 12,117
- Part-time: 9,696
- Temporary/Intern/Other: < 3,000

*Experience Levels:*
- Mid-Senior level: 41,489
- Entry level: 36,708
- Associate: 9,826
- Director/Executive: ~5,000

*Remote Allowed:* 15,246 postings (approx 12.3%) are explicitly marked as remote.

## 4. Conclusion
Milestone 1.5 is fully complete. 
- The tests are passing.
- The `01_data_contract.md` has been expanded and documented.
- The pipeline effortlessly ingested 123k postings, performed deduplicating aggregations, and produced a rich silver dataset ready for downstream analysis or ML. No LLMs, GPUs, or internet access were required.
