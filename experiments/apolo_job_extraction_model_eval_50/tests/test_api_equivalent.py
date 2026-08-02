"""Deterministic, network-free tests for OpenRouter-equivalent cost/timing
estimates on subscription/manual and recovered-historical runs."""

import hashlib
import json

from benchmarking import api_equivalent, historical_baselines, loaders, metrics
import compare_models


def _subscription_manifest(**overrides):
    m = {
        "n_rows_requested": 50, "n_rows_attempted": 50, "n_rows_completed": 50,
        "generation_method": "manual_first_party_model_execution",
        "total_input_tokens": 95179, "total_output_tokens": 17528, "total_tokens": 112707,
    }
    m.update(api_equivalent.full_estimate(95179, 17528, "gpt-5.5", 50, 50, 50))
    m.update(overrides)
    return m


# 1. Measured API timing has priority over an estimate.
def test_measured_timing_beats_estimate():
    m = _subscription_manifest(provider_measured_latency_seconds=2.5)
    assert metrics._speed_seconds(m) == (2.5, "measured")
    assert compare_models.speed_seconds(m) == 2.5


# 2. Subscription run with valid OpenRouter timing metadata is in speed reports.
def test_subscription_run_included_in_speed():
    m = _subscription_manifest()
    seconds, basis = metrics._speed_seconds(m)
    assert seconds is not None and basis == "openrouter_equivalent_estimate"
    assert compare_models.speed_seconds(m) == m["estimated_seconds_per_completed_job"]
    eff = metrics.efficiency_metrics(m)
    assert eff["seconds_per_valid_job"] is not None
    assert eff["completed_jobs_per_minute"] is not None


# 3. An estimate is never written to provider_measured_latency_seconds.
def test_estimate_never_in_provider_measured_field():
    block = api_equivalent.full_estimate(95179, 17528, "gpt-5.5", 50, 50, 50)
    assert block["provider_measured_latency_seconds"] is None
    assert block["timing_is_measured"] is False


# 4. observed_cost_usd remains null for subscription/manual runs.
def test_observed_cost_null_for_subscription():
    eff = metrics.efficiency_metrics(_subscription_manifest())
    assert eff["observed_cost_usd"] is None
    assert eff["estimated_cost_usd"] is not None
    assert eff["cost_basis"] == "openrouter_list_price_estimate"


# 5 & 6. 1.20 applied exactly once to time; never to cost.
def test_safety_factor_applied_once_to_time_not_cost():
    b = api_equivalent.full_estimate(95179, 17528, "gpt-5.5", 50, 50, 50)
    base = 50 * 3.23 + 17528 / 39
    assert abs(b["base_estimated_runtime_seconds"] - base) < 1e-9
    assert abs(b["estimated_inference_time_seconds"] - base * 1.20) < 1e-9
    # not base * 1.20 * 1.20
    assert abs(b["estimated_inference_time_seconds"] - base * 1.44) > 1.0
    cost_no_factor = 95179 / 1e6 * 5.0 + 17528 / 1e6 * 30.0
    assert abs(b["estimated_cost_usd"] - cost_no_factor) < 1e-12


# 7. Token-count fallback is deterministic.
def test_token_fallback_deterministic():
    assert api_equivalent.utf8_bytes_div4("abcd") == 1
    assert api_equivalent.utf8_bytes_div4("abcde") == 2
    s = "café ñ 日本"
    assert api_equivalent.utf8_bytes_div4(s) == api_equivalent.utf8_bytes_div4(s)


# 8 & 9. Historical row counts use actual recovered data; non-comparable
# Gemini excluded from frozen-sample quality but included in efficiency.
def test_historical_uses_actual_row_count_and_is_efficiency_only(tmp_path, monkeypatch):
    rows = [
        {"job_id": "1", "title": "t", "apolo_extraction": {"skills": [], "schema_version": "0.1.0"}},
        {"job_id": "2", "title": "t", "apolo_extraction": {"skills": [], "schema_version": "0.1.0"}},
        {"job_id": "3", "title": "t", "apolo_extraction": {"skills": [], "schema_version": "0.1.0"}},
    ]
    path = tmp_path / "hist.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    monkeypatch.setattr(historical_baselines, "HISTORICAL_JSONL", path)
    frozen_sample = {"1": {"job_id": "1", "title": "t", "job_card_text": "x" * 400, "description_text": "y" * 300}}

    rec = historical_baselines.historical_efficiency_report(["1"], frozen_sample)
    assert rec["n_rows"] == 3  # actual recovered rows, not 50
    assert rec["directly_comparable"] is False
    assert rec["quality_comparable_to_frozen_50"] is False
    assert rec["estimated_cost_usd"] is not None
    assert rec["estimated_seconds_per_completed_job"] is not None
    assert rec["timing_basis"] == "openrouter_equivalent_estimate"
    # quality is not reported for the historical efficiency record
    assert "quality_value" not in rec
    # the quality baseline report keeps directly_comparable False
    base = historical_baselines.build_baseline_report(loaders.load_frozen_job_ids())
    assert base["directly_comparable"] is False


# 10. All three targets have non-null estimated timing and cost.
def test_three_targets_non_null_estimates():
    for ti, to, key in [(95179, 17528, "gpt-5.5"), (95199, 12818, "claude-sonnet-5"),
                        (1000, 2000, "gemini-3.5-flash")]:
        b = api_equivalent.full_estimate(ti, to, key, 50, 50, 50)
        assert b["estimated_cost_usd"] > 0
        assert b["estimated_inference_time_seconds"] > 0
        assert b["estimated_seconds_per_completed_job"] > 0


# 11. Existing model extractions are untouched by the estimate pipeline.
def test_existing_extractions_unchanged():
    p = loaders.OUTPUTS_DIR / "openai__gpt-5.5__high" / "extractions.jsonl"
    before = hashlib.sha256(p.read_bytes()).hexdigest()
    api_equivalent.full_estimate(95179, 17528, "gpt-5.5", 50, 50, 50)
    metrics.efficiency_metrics(_subscription_manifest())
    assert hashlib.sha256(p.read_bytes()).hexdigest() == before


# 12. Generated JSON/JSONL parse.
def test_generated_manifests_and_reports_parse():
    for slug in ("openai__gpt-5.5__high", "anthropic__claude-sonnet-5"):
        json.loads((loaders.OUTPUTS_DIR / slug / "manifest.json").read_text(encoding="utf-8"))
    bench = loaders.REPORTS_DIR / "benchmark_table.json"
    if bench.exists():
        json.loads(bench.read_text(encoding="utf-8"))
