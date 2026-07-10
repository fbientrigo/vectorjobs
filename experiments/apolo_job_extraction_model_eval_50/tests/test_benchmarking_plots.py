import hashlib

from benchmarking import loaders, plots
from build_benchmark_report import build_model_row


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def one_model_row(**overrides):
    manifest = {
        "n_rows_requested": 50, "n_rows_completed": 46,
        "request_failures": 0, "json_parse_failures": 0, "schema_failures": 0,
        "evidence_substring_failures": 4, "empty_skill_rows": 0,
        "broad_domain_skill_rate": 0.4, "confidence_1_rate": 0.05,
        "estimated_cost_usd": 0.02, "wall_time_seconds": 60.0,
        "prompt_tokens": 1000, "completion_tokens": 2000, "total_tokens": 3000,
        "finished_at": "t",
    }
    manifest.update(overrides)
    return build_model_row("fake/model-a", manifest, gold_row=None, n_jobs_target=50)


def test_plotting_works_with_one_model_without_gold(tmp_path):
    rows = [one_model_row()]
    for plot_fn in plots.ALL_PLOTS:
        paths, notes = plot_fn(rows, tmp_path)
        # Either it produced files, or it explained why not — never a crash.
        assert paths or notes

    for row in rows:
        assert row["quality_quality_source"] == "operational_quality_proxy"


def test_plotting_with_unavailable_historical_cost_latency(tmp_path):
    rows = [one_model_row()]
    historical = {
        "display_name": "Gemini 3.5 Flash historical baseline",
        "cost_available": False,
        "latency_available": False,
        "evidence_grounded_rate": 0.9,
    }
    paths, notes = plots.plot_reliability_vs_quality(rows, tmp_path, historical=historical)
    assert notes  # historical omitted with an explanation, not silently zeroed
    assert all("omitted" in n for n in notes)


def test_retry_burden_skips_cleanly_when_no_attempt_data(tmp_path):
    rows = [one_model_row()]
    paths, notes = plots.plot_model_retry_burden(rows, tmp_path)
    assert paths == []
    assert notes


def test_run_outputs_remain_unchanged_after_plotting(tmp_path):
    manifest_path = loaders.OUTPUTS_DIR / "deepseek__deepseek-v4-flash" / "manifest.json"
    before = sha256_file(manifest_path)

    rows = [one_model_row()]
    for plot_fn in plots.ALL_PLOTS:
        plot_fn(rows, tmp_path)

    assert sha256_file(manifest_path) == before
