"""Family 1: operational reliability. Always evaluated over all n_jobs_target
frozen jobs (cohort A) — never restricted to a successful-job intersection.

run_model_eval.py (the current runner) retries transport failures internally
inside call_openrouter() but does not persist per-attempt records — only the
aggregate outcome per job survives in manifest.json. So for the five models
already run, "first pass" vs. "eventual" cannot be distinguished from the
retry-recovery case; we report a single-run completion measure
(single_run_valid_count / single_run_valid_rate) and leave every
attempt-level field (n_valid_first_pass, total_retry_attempts, ...) null
rather than inventing a number.

attempt_log is an optional escape hatch for a future resume/recovery segment
that DOES persist attempt-level counts (see "No automatic recovery" in the
task spec). When supplied, first-pass and eventual metrics are computed
independently — the eventual segment never overwrites the first-pass one.
"""

RUN_STATUS_VALID_ALL = "process_finished_valid_50"
RUN_STATUS_PARTIAL = "process_finished_with_invalid_or_missing_rows"
RUN_STATUS_INTERRUPTED = "interrupted"
RUN_STATUS_BUDGET_STOPPED = "budget_stopped"
RUN_STATUS_INCOMPLETE = "incomplete_execution"


def run_status(manifest: dict, n_jobs_target: int) -> str:
    if not manifest or not manifest.get("finished_at"):
        return RUN_STATUS_INTERRUPTED
    n_requested = manifest.get("n_rows_requested") or 0
    if n_requested < n_jobs_target:
        return RUN_STATUS_BUDGET_STOPPED
    n_completed = manifest.get("n_rows_completed") or 0
    if n_completed == n_jobs_target:
        return RUN_STATUS_VALID_ALL
    return RUN_STATUS_PARTIAL


def _rate(numerator, denominator):
    if numerator is None or not denominator:
        return None
    return numerator / denominator


def reliability_metrics(manifest: dict, n_jobs_target: int, attempt_log: dict | None = None) -> dict:
    manifest = manifest or {}
    n_jobs_attempted = manifest.get("n_rows_requested") or n_jobs_target
    single_run_valid_count = manifest.get("n_rows_completed") or 0
    single_run_valid_rate = _rate(single_run_valid_count, n_jobs_target)
    n_failed_final_single_run = n_jobs_attempted - single_run_valid_count

    metrics = {
        "n_jobs_target": n_jobs_target,
        "n_jobs_attempted": n_jobs_attempted,
        "n_valid_first_pass": None,
        "first_pass_valid_rate": None,
        "n_eventually_valid": single_run_valid_count,
        "eventual_valid_rate": single_run_valid_rate,
        "n_failed_first_pass": None,
        "n_failed_final": n_failed_final_single_run,
        "total_request_attempts": None,
        "total_retry_attempts": None,
        "retry_rate": None,
        "retries_per_target_job": None,
        "attempts_per_valid_job": None,
        "retry_recovered_jobs": None,
        "retry_recovery_rate": None,
        "permanent_failure_rate": _rate(n_failed_final_single_run, n_jobs_target),
        "single_run_valid_count": single_run_valid_count,
        "single_run_valid_rate": single_run_valid_rate,
        "attempt_level_data_available": False,
        "run_status": run_status(manifest, n_jobs_target),
        "note": (
            "single_run_valid_count/rate is a single-run completion measure, "
            "not proven first-attempt success; run_model_eval.py does not "
            "persist per-attempt records (see module docstring)."
        ),
    }

    if not attempt_log:
        return metrics

    n_valid_first_pass = attempt_log["initial_segment_valid_count"]
    n_eventually_valid = attempt_log.get("cumulative_valid_count", n_valid_first_pass)
    total_request_attempts = attempt_log.get("total_request_attempts")
    total_retry_attempts = attempt_log.get("total_retry_attempts")
    retry_recovered_jobs = attempt_log.get("retry_recovered_jobs")
    n_failed_first_pass = n_jobs_target - n_valid_first_pass
    n_failed_final = n_jobs_target - n_eventually_valid

    metrics.update({
        "n_valid_first_pass": n_valid_first_pass,
        "first_pass_valid_rate": _rate(n_valid_first_pass, n_jobs_target),
        "n_eventually_valid": n_eventually_valid,
        "eventual_valid_rate": _rate(n_eventually_valid, n_jobs_target),
        "n_failed_first_pass": n_failed_first_pass,
        "n_failed_final": n_failed_final,
        "total_request_attempts": total_request_attempts,
        "total_retry_attempts": total_retry_attempts,
        "retry_rate": _rate(total_retry_attempts, total_request_attempts),
        "retries_per_target_job": _rate(total_retry_attempts, n_jobs_target),
        "attempts_per_valid_job": _rate(total_request_attempts, n_eventually_valid),
        "retry_recovered_jobs": retry_recovered_jobs,
        "retry_recovery_rate": _rate(retry_recovered_jobs, n_failed_first_pass),
        "permanent_failure_rate": _rate(n_failed_final, n_jobs_target),
        "attempt_level_data_available": True,
    })
    return metrics
