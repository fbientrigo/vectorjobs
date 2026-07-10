"""Three comparison cohorts. See build_benchmark_report.py / methodology doc.

A. Full-target reliability cohort  - all 50 frozen job_ids, always.
B. Common semantic cohort          - intersection of successfully-extracted
                                      job_ids among the models being compared.
C. Gold-reviewed cohort            - job_ids with review_status != "pending"
                                      in a reviewed gold file.

Reliability metrics must always use cohort A. Never substitute B for A.
"""


def full_target_cohort(frozen_job_ids: list[str]) -> set[str]:
    return set(str(j) for j in frozen_job_ids)


def common_semantic_cohort(model_job_ids: dict[str, set[str]]) -> tuple[set[str], dict]:
    """Intersection of successfully-extracted job_ids across the given models.

    model_job_ids: {model_name: set of job_ids that model produced a valid
    extraction for}. Empty input returns an empty cohort, not an error.
    """
    if not model_job_ids:
        return set(), {"n_common_jobs": 0, "models": []}
    common = set.intersection(*model_job_ids.values())
    return common, {"n_common_jobs": len(common), "models": sorted(model_job_ids)}


def gold_reviewed_cohort(gold_rows: list[dict]) -> set[str]:
    return {str(row["job_id"]) for row in gold_rows if row.get("review_status") != "pending"}
