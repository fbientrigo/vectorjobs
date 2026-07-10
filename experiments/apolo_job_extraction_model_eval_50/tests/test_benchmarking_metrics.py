from benchmarking import cohorts, metrics


def test_failed_requests_still_contribute_to_total_observed_cost():
    # Upstream run_model_eval.py accumulates cost_delta for every response,
    # success or failure. Our layer must surface that total, not silently
    # drop the failed-row share of it.
    manifest = {
        "n_rows_requested": 2, "n_rows_completed": 1, "request_failures": 1,
        "estimated_cost_usd": 0.5, "wall_time_seconds": 10.0,
    }
    eff = metrics.efficiency_metrics(manifest)
    assert eff["observed_cost_usd"] == 0.5
    assert eff["cost_per_attempted_job"] == 0.25


def test_cost_per_valid_job_uses_valid_output_denominator():
    manifest = {"n_rows_requested": 2, "n_rows_completed": 1, "estimated_cost_usd": 0.5}
    eff = metrics.efficiency_metrics(manifest)
    assert eff["cost_per_eventually_valid_job"] == 0.5
    assert eff["cost_per_first_pass_valid_job"] == 0.5


def test_missing_cost_stays_null_not_zero():
    manifest = {"n_rows_requested": 2, "n_rows_completed": 2}
    eff = metrics.efficiency_metrics(manifest)
    assert eff["observed_cost_usd"] is None
    assert eff["cost_per_attempted_job"] is None
    assert eff["cost_source"] is None


def test_missing_latency_stays_null_not_zero():
    manifest = {"n_rows_requested": 2, "n_rows_completed": 2}
    eff = metrics.efficiency_metrics(manifest)
    assert eff["wall_time_seconds"] is None
    assert eff["seconds_per_valid_job"] is None
    assert eff["completed_jobs_per_minute"] is None


def test_structural_correctness_separates_failure_categories():
    manifest = {
        "n_rows_requested": 5, "n_rows_completed": 1,
        "request_failures": 1, "json_parse_failures": 1, "schema_failures": 1,
        "evidence_substring_failures": 1,
    }
    structural = metrics.structural_correctness(manifest)
    assert structural["n_request_failures"] == 1
    assert structural["n_json_parse_failures"] == 1
    assert structural["n_schema_failures"] == 1
    assert structural["n_evidence_substring_failures"] == 1
    assert structural["n_missing_final_rows"] == 4


def test_semantic_quality_without_gold_uses_labelled_proxy():
    manifest = {"n_rows_completed": 10, "empty_skill_rows": 0, "broad_domain_skill_rate": 0.3, "confidence_1_rate": 0.0}
    quality = metrics.semantic_quality("m", manifest, gold_row=None, n_jobs_target=50)
    assert quality["quality_source"] == "operational_quality_proxy"
    assert quality["skill_strict_type_f1"] is None
    assert 0.0 <= quality["quality_value"] <= 1.0


def test_semantic_quality_prefers_gold_when_available():
    manifest = {"n_rows_completed": 10}
    gold_row = {"n_gold_reviewed": 50, "skill_strict_type_f1": 0.7}
    quality = metrics.semantic_quality("m", manifest, gold_row=gold_row, n_jobs_target=50)
    assert quality["quality_source"] == "human_gold"
    assert quality["quality_value"] == 0.7


def test_semantic_quality_reviewed_subset_when_gold_partial():
    manifest = {"n_rows_completed": 10}
    gold_row = {"n_gold_reviewed": 12, "skill_strict_type_f1": 0.6}
    quality = metrics.semantic_quality("m", manifest, gold_row=gold_row, n_jobs_target=50)
    assert quality["quality_source"] == "reviewed_subset_gold"


def test_common_semantic_cohort_uses_only_shared_successful_job_ids():
    model_job_ids = {
        "a": {"1", "2", "3"},
        "b": {"2", "3", "4"},
    }
    common, info = cohorts.common_semantic_cohort(model_job_ids)
    assert common == {"2", "3"}
    assert info["n_common_jobs"] == 2
