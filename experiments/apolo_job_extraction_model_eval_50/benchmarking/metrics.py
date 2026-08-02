"""Families 2 (structural correctness), 3 (semantic quality), 4 (efficiency).

Every rate carries an explicit denominator field alongside it. Missing
inputs (no cost reported, no gold reviewed) stay None — never coerced to 0.
"""


def _rate(numerator, denominator):
    if numerator is None or not denominator:
        return None
    return numerator / denominator


def _safe_div(numerator, denominator):
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _speed_seconds(manifest: dict):
    """Seconds per completed job with an explicit basis, never local wall time.

    Priority: measured provider latency > OpenRouter-equivalent estimate >
    measured per-job API latency (real OpenRouter runs). Manual/subscription
    runs without a defensible estimate stay None. Returns (seconds, basis).
    """
    if manifest.get("provider_measured_latency_seconds") is not None:
        return manifest.get("provider_measured_latency_seconds"), "measured"
    if manifest.get("estimated_seconds_per_completed_job") is not None:
        return manifest.get("estimated_seconds_per_completed_job"), "openrouter_equivalent_estimate"
    generation_method = str(manifest.get("generation_method") or "").lower()
    limitations = " ".join(str(v).lower() for v in manifest.get("limitations", []))
    if "manual" in generation_method or "no external api" in limitations or "no per-job api call" in limitations:
        return None, None
    latency = manifest.get("latency_seconds")
    return latency, ("measured" if latency is not None else None)


def _cost_reporting(manifest: dict):
    """(observed, estimated, effective, basis, source). Subscription runs keep
    observed=None and carry the OpenRouter list-price estimate."""
    value = manifest.get("estimated_cost_usd")
    if manifest.get("cost_basis") == "openrouter_list_price_estimate" or manifest.get("cost_is_measured") is False:
        estimated = manifest.get("estimated_cost_usd")
        return None, estimated, estimated, "openrouter_list_price_estimate", "openrouter_list_price_estimate"
    if value is None:
        return None, None, None, None, None
    return value, None, value, "measured", "openrouter_usage_field"


def structural_correctness(manifest: dict) -> dict:
    """Separates transport/JSON/schema/validation/evidence failures.

    All rates are out of n_attempted (== n_rows_requested), except
    empty_skill_row_rate, whose denominator is n_completed (only completed
    rows have a skills list to be empty or not).
    """
    manifest = manifest or {}
    n_attempted = manifest.get("n_rows_requested") or 0
    n_completed = manifest.get("n_rows_completed") or 0
    request_failures = manifest.get("request_failures") or 0
    empty_output_failures = manifest.get("empty_output_failures") or 0
    json_parse_failures = manifest.get("json_parse_failures") or 0
    schema_failures = manifest.get("schema_failures") or 0
    evidence_substring_failures = manifest.get(
        "evidence_substring_failures", manifest.get("validation_failure_count", 0)
    ) or 0
    empty_skill_rows = manifest.get("empty_skill_rows") or 0

    return {
        "n_attempted": n_attempted,
        "n_completed": n_completed,
        "request_success_rate": _rate(n_attempted - request_failures, n_attempted),
        "json_valid_rate": _rate(
            n_attempted - request_failures - empty_output_failures - json_parse_failures, n_attempted
        ),
        "schema_valid_rate": _rate(
            n_attempted - request_failures - empty_output_failures - json_parse_failures - schema_failures,
            n_attempted,
        ),
        "validation_success_rate": _rate(n_completed, n_attempted),
        "evidence_grounded_rate": _rate(n_attempted - evidence_substring_failures, n_attempted),
        "empty_skill_row_rate": _rate(empty_skill_rows, n_completed),
        "n_request_failures": request_failures,
        "n_empty_output_failures": empty_output_failures,
        "n_json_parse_failures": json_parse_failures,
        "n_schema_failures": schema_failures,
        "n_evidence_substring_failures": evidence_substring_failures,
        "n_missing_final_rows": n_attempted - n_completed,
    }


def operational_quality_proxy(manifest: dict) -> float | None:
    """Heuristic in [0, 1] from structural/diagnostic signals only.

    NOT an accuracy score. Excludes cost and speed (those stay in family 4)
    and does not reward extracting more skills. Use only when no gold —
    reviewed or partial — exists for this model (see semantic_quality()).
    """
    manifest = manifest or {}
    n_completed = manifest.get("n_rows_completed") or 0
    if n_completed == 0:
        return None
    empty_rate = (manifest.get("empty_skill_rows") or 0) / n_completed
    broad_rate = manifest.get("broad_domain_skill_rate") or 0.0
    conf1_rate = manifest.get("confidence_1_rate") or 0.0

    score = 1.0
    score -= 0.4 * empty_rate
    score -= 0.3 * max(0.0, broad_rate - 0.5)
    score -= 0.3 * max(0.0, conf1_rate - 0.2)
    return max(0.0, min(1.0, score))


def semantic_quality(model: str, manifest: dict, gold_row: dict | None, n_jobs_target: int) -> dict:
    """Hierarchy: human gold > reviewed subset > operational_quality_proxy.

    gold_row is one row from evaluate_against_gold.py's model_vs_gold.csv
    (via loaders.load_gold_vs_model), or None if that script hasn't run /
    found no reviewed gold yet.
    """
    n_gold_reviewed = (gold_row or {}).get("n_gold_reviewed") or 0
    if gold_row and n_gold_reviewed > 0:
        source = "human_gold" if n_gold_reviewed >= n_jobs_target else "reviewed_subset_gold"
        return {
            "quality_source": source,
            "n_gold_reviewed": n_gold_reviewed,
            "quality_value": gold_row.get("skill_strict_type_f1"),
            "skill_name_exact_f1": gold_row.get("skill_name_exact_f1"),
            "skill_name_normalized_f1": gold_row.get("skill_name_normalized_f1"),
            "skill_strict_type_f1": gold_row.get("skill_strict_type_f1"),
            "education_exact_f1": gold_row.get("education_exact_f1"),
            "seniority_accuracy": gold_row.get("seniority_accuracy"),
        }

    return {
        "quality_source": "operational_quality_proxy",
        "n_gold_reviewed": 0,
        "quality_value": operational_quality_proxy(manifest),
        "skill_name_exact_f1": None,
        "skill_name_normalized_f1": None,
        "skill_strict_type_f1": None,
        "education_exact_f1": None,
        "seniority_accuracy": None,
    }


def efficiency_metrics(manifest: dict) -> dict:
    """Cost/latency stay None (not 0) when the runner never observed them.

    inference_time_seconds and retry_wait_time_seconds are None because
    run_model_eval.py's per-job latency includes any internal retry backoff
    sleep inside call_openrouter() — the two cannot be separated post hoc.
    """
    manifest = manifest or {}
    n_attempted = manifest.get("n_rows_requested") or 0
    n_completed = manifest.get("n_rows_completed") or 0
    observed_cost, estimated_cost, cost, cost_basis, cost_source = _cost_reporting(manifest)
    wall_time = manifest.get("wall_time_seconds")
    speed_seconds, timing_basis = _speed_seconds(manifest)

    return {
        "observed_cost_usd": observed_cost,
        "estimated_cost_usd": estimated_cost,
        "cost_for_reporting_usd": cost,
        "cost_basis": cost_basis,
        "cost_source": cost_source,
        "total_input_tokens": manifest.get("total_input_tokens", manifest.get("prompt_tokens")),
        "total_output_tokens": manifest.get("total_output_tokens", manifest.get("completion_tokens")),
        "total_reasoning_tokens": None,
        "total_tokens": manifest.get("total_tokens"),
        "wall_time_seconds": wall_time,
        "inference_time_seconds": manifest.get("estimated_inference_time_seconds"),
        "estimated_inference_time_seconds": manifest.get("estimated_inference_time_seconds"),
        "retry_wait_time_seconds": None,
        "timing_basis": timing_basis,
        "latency_safety_factor": manifest.get("latency_safety_factor"),
        "completed_jobs_per_minute": _safe_div(60, speed_seconds),
        "seconds_per_attempted_job": manifest.get("estimated_seconds_per_attempted_job")
            if timing_basis == "openrouter_equivalent_estimate" else _safe_div(wall_time, n_attempted),
        "seconds_per_valid_job": speed_seconds,
        "cost_per_attempted_job": _safe_div(cost, n_attempted),
        "cost_per_first_pass_valid_job": _safe_div(cost, n_completed),
        "cost_per_eventually_valid_job": _safe_div(cost, n_completed),
        "API_calls_per_valid_job": _safe_div(n_attempted, n_completed),
    }
