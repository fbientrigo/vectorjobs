import json
from pathlib import Path

import numpy as np
import pandas as pd
from click.testing import CliRunner

from jobsrec.cli import main
from jobsrec.trends.temporal import (
    add_month_bucket,
    build_reliability_assessment,
    compute_annual_salary,
    compute_centroid_drift,
    compute_salary_weighted_centroid_drift,
    compute_semantic_centroid_drift,
    compute_skill_growth,
    compute_temporal_audit,
    compute_temporal_column_coverage,
    parse_listed_time,
    prepare_salary_coverage_plot_data,
    prepare_support_aware_drift_plot_data,
    prepare_salary_weights,
    sample_jobs,
)

from scripts.build_presentation_assets import GENERATED_FIGURES, REQUIRED_FIGURES


def _synthetic_jobs(n_months: int = 6, rows_per_month: int = 3) -> pd.DataFrame:
    rows = []
    job_id = 1
    for month in range(1, n_months + 1):
        for offset in range(rows_per_month):
            early_skill = "excel" if month <= n_months // 2 else "python"
            rows.append(
                {
                    "job_id": job_id,
                    "title": f"Role {job_id}",
                    "description": "Build data products",
                    "job_card_text": f"data analytics month {month} {early_skill}",
                    "skills_text": f"sql; {early_skill}",
                    "listed_time": f"2024-{month:02d}-{offset + 1:02d}",
                    "original_listed_time": f"2023-{month:02d}-{offset + 1:02d}" if month <= 2 else f"2024-{month:02d}-{offset + 1:02d}",
                    "expiry": f"2024-{month:02d}-{min(offset + 10, 28):02d}",
                    "closed_time": None,
                    "normalized_salary": 50000.0 + (job_id * 1000),
                    "min_salary": 40000.0 + (job_id * 1000),
                    "max_salary": 60000.0 + (job_id * 1000),
                    "med_salary": None,
                    "pay_period": "YEARLY",
                    "currency": "USD",
                    "compensation_type": "BASE_SALARY",
                }
            )
            job_id += 1
    return pd.DataFrame(rows)


def test_month_parsing_works() -> None:
    df = pd.DataFrame({"listed_time": ["2024-01-15"]})

    parsed = parse_listed_time(df)
    bucketed = add_month_bucket(df)

    assert parsed.notna().all()
    assert bucketed.loc[0, "month"] == "2024-01"


def test_numeric_epoch_milliseconds_parse_to_real_month() -> None:
    df = pd.DataFrame({"listed_time": [1713139200000]})

    bucketed = add_month_bucket(df)

    assert bucketed.loc[0, "month"] == "2024-04"


def test_invalid_dates_are_excluded_from_temporal_sampling() -> None:
    df = _synthetic_jobs(n_months=2, rows_per_month=1)
    df.loc[1, "listed_time"] = "not-a-date"

    sampled = sample_jobs(df, sample_size=10, sampling_mode="temporal-stride")

    assert len(sampled) == 1
    assert sampled["_listed_time_parsed"].notna().all()


def test_temporal_stride_sampling_covers_first_and_last_month() -> None:
    df = _synthetic_jobs(n_months=12, rows_per_month=2)

    sampled = sample_jobs(df, sample_size=5, sampling_mode="temporal-stride")

    assert sampled["month"].iloc[0] == "2024-01"
    assert sampled["month"].iloc[-1] == "2024-12"


def test_temporal_stride_sampling_preserves_each_month_when_possible() -> None:
    df = _synthetic_jobs(n_months=5, rows_per_month=5)

    sampled = sample_jobs(df, sample_size=5, sampling_mode="temporal-stride")

    assert sampled["month"].tolist() == ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05"]


def test_random_sampling_is_deterministic() -> None:
    df = _synthetic_jobs(n_months=8, rows_per_month=3)

    first = sample_jobs(df, sample_size=7, sampling_mode="random")
    second = sample_jobs(df, sample_size=7, sampling_mode="random")

    assert first["job_id"].tolist() == second["job_id"].tolist()


def test_centroid_drift_output_has_required_columns() -> None:
    df = add_month_bucket(_synthetic_jobs(n_months=3, rows_per_month=2))
    vectors = pd.get_dummies(df["job_id"]).to_numpy()

    drift = compute_centroid_drift(df, vectors)

    assert set(["month", "previous_month", "cosine_similarity", "centroid_drift", "jobs_in_month"]).issubset(
        drift.columns
    )
    assert len(drift) == 3


def test_skill_growth_detects_rising_and_declining_skills() -> None:
    df = add_month_bucket(_synthetic_jobs(n_months=4, rows_per_month=3))

    growth = compute_skill_growth(df)

    python_delta = growth.loc[growth["skill"] == "python", "share_delta"].iloc[0]
    excel_delta = growth.loc[growth["skill"] == "excel", "share_delta"].iloc[0]
    assert python_delta > 0
    assert excel_delta < 0


def test_temporal_demo_cli_writes_required_outputs(tmp_path: Path) -> None:
    silver_path = tmp_path / "jobs.parquet"
    output_dir = tmp_path / "gold"
    figures_dir = tmp_path / "figures"
    report_path = tmp_path / "report.md"
    _synthetic_jobs(n_months=5, rows_per_month=3).to_parquet(silver_path, index=False)

    result = CliRunner().invoke(
        main,
        [
            "temporal-demo",
            "--silver-path",
            str(silver_path),
            "--output-dir",
            str(output_dir),
            "--figures-dir",
            str(figures_dir),
            "--report-path",
            str(report_path),
            "--sample-size",
            "12",
            "--sampling-mode",
            "temporal-stride",
        ],
    )

    assert result.exit_code == 0, result.output
    assert report_path.exists()
    assert (output_dir / "temporal_manifest.json").exists()
    assert (output_dir / "monthly_centroid_drift.parquet").exists()
    assert (output_dir / "skill_growth.parquet").exists()
    assert (figures_dir / "job_volume_by_month.png").exists()
    assert (figures_dir / "centroid_drift_by_month.png").exists()
    assert (figures_dir / "top_rising_skills.png").exists()
    assert (figures_dir / "top_declining_skills.png").exists()
    assert (figures_dir / "centroid_drift_by_month.png").stat().st_size > 0


def test_temporal_audit_cli_writes_schema_outputs(tmp_path: Path) -> None:
    silver_path = tmp_path / "jobs.parquet"
    output_dir = tmp_path / "audit"
    df = _synthetic_jobs(n_months=2, rows_per_month=2)
    df.loc[0, "listed_time"] = "not-a-date"
    df.to_parquet(silver_path, index=False)

    result = CliRunner().invoke(
        main,
        [
            "temporal-audit",
            "--input",
            str(silver_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    summary = json.loads((output_dir / "summary.json").read_text())
    monthly = pd.read_parquet(output_dir / "monthly_counts.parquet")
    weekly = pd.read_parquet(output_dir / "weekly_counts.parquet")
    daily = pd.read_parquet(output_dir / "daily_counts.parquet")
    temporal_coverage = pd.read_parquet(output_dir / "temporal_column_coverage.parquet")
    assert summary["total_rows"] == 4
    assert summary["valid_listed_time_rows"] == 3
    assert summary["invalid_or_missing_listed_time_rows"] == 1
    assert 0 < summary["parse_success_rate"] < 1
    assert summary["number_of_months"] == 2
    assert {"month", "rows", "job_card_text_coverage", "skills_text_coverage"}.issubset(monthly.columns)
    assert {"week", "rows"}.issubset(weekly.columns)
    assert {"day", "rows"}.issubset(daily.columns)
    assert {"time_column", "number_of_months", "reliability_label"}.issubset(temporal_coverage.columns)
    assert (output_dir / "report.md").exists()


def test_compute_temporal_audit_reports_coverage_by_month() -> None:
    df = _synthetic_jobs(n_months=2, rows_per_month=2)
    df.loc[0, "job_card_text"] = ""
    df.loc[1, "skills_text"] = ""

    summary, monthly, _, _, coverage = compute_temporal_audit(df, input_path="memory")

    assert summary["total_rows"] == 4
    assert monthly.loc[monthly["month"] == "2024-01", "job_card_text_coverage"].iloc[0] == 0.5
    assert monthly.loc[monthly["month"] == "2024-01", "skills_text_coverage"].iloc[0] == 0.5
    assert summary["reliability_label"] == "limited_temporal_coverage"
    assert {"listed_time", "original_listed_time"}.issubset(set(coverage["time_column"]))


def test_reliability_gate_labels_two_month_dataset() -> None:
    assessment = build_reliability_assessment({"2024-03": 5000, "2024-04": 5000})

    assert assessment["label"] == "limited_temporal_coverage"
    assert any("two-bucket comparison" in warning for warning in assessment["warnings"])


def test_reliability_gate_warns_when_month_under_1000_rows() -> None:
    assessment = build_reliability_assessment({"2024-01": 999, "2024-02": 1500, "2024-03": 1500})

    assert assessment["label"] == "limited_temporal_coverage"
    assert assessment["low_support_months"] == {"2024-01": 999}
    assert any("below 1000 rows" in warning for warning in assessment["warnings"])


def test_reliability_gate_sufficient_for_six_supported_months() -> None:
    assessment = build_reliability_assessment({f"2024-{month:02d}": 1000 for month in range(1, 7)})

    assert assessment["label"] == "sufficient_temporal_coverage"
    assert assessment["warnings"] == []


def test_semantic_centroid_drift_outputs_required_columns(tmp_path: Path) -> None:
    df = add_month_bucket(_synthetic_jobs(n_months=3, rows_per_month=2))
    embeddings = np.eye(len(df), 6, dtype=np.float32)

    drift, metadata = compute_semantic_centroid_drift(
        df,
        embeddings,
        embedding_backend="mock",
        embedding_model="deterministic-mock",
        centroids_path=tmp_path / "centroids.npy",
    )

    assert {
        "month_from",
        "month_to",
        "n_from",
        "n_to",
        "representation",
        "embedding_backend",
        "embedding_model",
        "cosine_similarity",
        "cosine_distance",
    }.issubset(drift.columns)
    assert len(drift) == 2
    assert {"month", "n_jobs", "vector_dim", "centroid_storage_path"}.issubset(metadata.columns)
    assert (tmp_path / "centroids.npy").exists()


def test_temporal_demo_semantic_mock_manifest_and_outputs(tmp_path: Path) -> None:
    silver_path = tmp_path / "jobs.parquet"
    output_dir = tmp_path / "semantic"
    _synthetic_jobs(n_months=3, rows_per_month=3).to_parquet(silver_path, index=False)

    result = CliRunner().invoke(
        main,
        [
            "temporal-demo",
            "--input",
            str(silver_path),
            "--output-dir",
            str(output_dir),
            "--sample-size",
            "9",
            "--representation",
            "semantic_embeddings",
            "--embedding-backend",
            "mock",
            "--embedding-model",
            "deterministic-mock",
            "--embedding-batch-size",
            "2",
            "--embedding-cache-dir",
            str(tmp_path / "cache"),
            "--device",
            "cpu",
            "--max-embedding-rows",
            "9",
        ],
    )

    assert result.exit_code == 0, result.output
    manifest = json.loads((output_dir / "temporal_manifest.json").read_text())
    drift = pd.read_parquet(output_dir / "monthly_centroid_drift.parquet")
    assert manifest["representation"] == "semantic_embeddings"
    assert manifest["embedding_backend"] == "mock"
    assert manifest["embedding_model"] == "deterministic-mock"
    assert manifest["embedding_batch_size"] == 2
    assert manifest["device"] == "cpu"
    assert manifest["embedding_cache_dir"] == str(tmp_path / "cache")
    assert manifest["embedding_cache_path"]
    assert manifest["reliability_label"] == "limited_temporal_coverage"
    assert (output_dir / "monthly_centroid_metadata.parquet").exists()
    assert (output_dir / "monthly_centroids.npy").exists()
    assert {"month_from", "month_to", "cosine_distance"}.issubset(drift.columns)


def test_original_listed_time_can_expand_temporal_buckets() -> None:
    df = pd.DataFrame(
        {
            "listed_time": ["2024-04-01", "2024-04-02", "2024-04-03"],
            "original_listed_time": ["2024-01-01", "2024-02-01", "2024-04-01"],
        }
    )

    listed = add_month_bucket(df, time_column="listed_time")
    original = add_month_bucket(df, time_column="original_listed_time")

    assert listed["month"].nunique() == 1
    assert original["month"].nunique() == 3


def test_temporal_column_coverage_reports_all_present_time_columns() -> None:
    df = _synthetic_jobs(n_months=3, rows_per_month=1)
    df.loc[0, "closed_time"] = "2024-04-05"

    coverage = compute_temporal_column_coverage(df)

    assert {"listed_time", "original_listed_time", "expiry", "closed_time"}.issubset(
        set(coverage["time_column"])
    )
    original_months = coverage.loc[coverage["time_column"] == "original_listed_time", "number_of_months"].iloc[0]
    assert original_months >= 3


def test_compute_annual_salary_prefers_normalized_then_annualizes_fallbacks() -> None:
    df = pd.DataFrame(
        {
            "normalized_salary": [100000.0, None, None],
            "med_salary": [999.0, 50.0, None],
            "min_salary": [None, None, 10.0],
            "max_salary": [None, None, 20.0],
            "pay_period": ["YEARLY", "HOURLY", "MONTHLY"],
        }
    )

    annual = compute_annual_salary(df)

    assert annual.iloc[0] == 100000.0
    assert annual.iloc[1] == 50.0 * 2080
    assert annual.iloc[2] == 15.0 * 12


def test_prepare_salary_weights_filters_non_usd_and_normalizes_by_month() -> None:
    df = add_month_bucket(_synthetic_jobs(n_months=1, rows_per_month=3))
    df.loc[0, "currency"] = "EUR"

    weights, diagnostics, summary = prepare_salary_weights(df)

    assert weights.notna().sum() == 2
    assert diagnostics.loc[0, "salary_usable"] == np.False_
    assert summary["salary_non_usd_excluded"] == 1
    np.testing.assert_allclose(weights.dropna().mean(), 1.0, rtol=1e-6)


def test_salary_weighted_centroid_drift_outputs_required_columns(tmp_path: Path) -> None:
    df = add_month_bucket(_synthetic_jobs(n_months=2, rows_per_month=2))
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    drift, metadata, diagnostics, summary, centroids_path = compute_salary_weighted_centroid_drift(
        df,
        vectors,
        representation="tfidf_svd",
        time_column="listed_time",
        output_dir=tmp_path,
    )

    assert {
        "month_from",
        "month_to",
        "n_from",
        "n_to",
        "n_salary_from",
        "n_salary_to",
        "salary_coverage_from",
        "salary_coverage_to",
        "representation",
        "time_column",
        "centroid_weighting",
        "cosine_similarity",
        "cosine_distance",
    }.issubset(drift.columns)
    assert len(drift) == 1
    assert len(metadata) == 2
    assert diagnostics["salary_weight"].notna().all()
    assert summary["salary_rows_used"] == 4
    assert centroids_path.exists()


def test_support_aware_drift_plot_data_flags_low_n_bins() -> None:
    drift = pd.DataFrame(
        {
            "month": ["2024-03-30", "2024-04-01"],
            "centroid_drift": [0.7, 0.1],
            "jobs_in_month": [1, 250],
        }
    )

    plot_df = prepare_support_aware_drift_plot_data(drift, low_support_threshold=100)

    assert plot_df.loc[0, "low_support"] == np.True_
    assert plot_df.loc[1, "low_support"] == np.False_
    assert plot_df["marker_size"].gt(0).all()


def test_salary_coverage_plot_data_exposes_support_and_overall_reference() -> None:
    metadata = pd.DataFrame(
        {
            "month": ["2024-02-01", "2024-04-01"],
            "n_jobs": [1, 9],
            "n_salary_jobs": [1, 2],
            "salary_coverage": [1.0, 2 / 9],
        }
    )

    plot_df, overall = prepare_salary_coverage_plot_data(metadata, low_support_threshold=5)

    assert plot_df.loc[0, "low_support"] == np.True_
    assert plot_df.loc[0, "salary_coverage"] == 1.0
    assert overall == 0.3


def test_temporal_demo_cli_time_column_and_salary_weighting_outputs(tmp_path: Path) -> None:
    silver_path = tmp_path / "jobs.parquet"
    output_dir = tmp_path / "weighted"
    df = _synthetic_jobs(n_months=4, rows_per_month=2)
    df["listed_time"] = "2024-04-01"
    df.to_parquet(silver_path, index=False)

    result = CliRunner().invoke(
        main,
        [
            "temporal-demo",
            "--input",
            str(silver_path),
            "--output-dir",
            str(output_dir),
            "--sample-size",
            "8",
            "--time-column",
            "original_listed_time",
            "--centroid-weighting",
            "both",
        ],
    )

    assert result.exit_code == 0, result.output
    manifest = json.loads((output_dir / "temporal_manifest.json").read_text())
    weighted = pd.read_parquet(output_dir / "monthly_centroid_drift_salary_weighted.parquet")
    assert manifest["time_column"] == "original_listed_time"
    assert manifest["centroid_weighting"] == "both"
    assert manifest["salary_rows_used"] == 8
    assert (output_dir / "monthly_centroid_metadata_salary_weighted.parquet").exists()
    assert (output_dir / "salary_weight_diagnostics.parquet").exists()
    assert (output_dir / "figures" / "centroid_drift_salary_weighted_by_month.png").exists()
    assert (output_dir / "figures" / "salary_coverage_by_month.png").exists()
    assert (output_dir / "figures" / "centroid_drift_salary_weighted_by_month.png").stat().st_size > 0
    assert (output_dir / "figures" / "salary_coverage_by_month.png").stat().st_size > 0
    assert {"n_salary_from", "n_salary_to", "salary_coverage_from", "salary_coverage_to"}.issubset(
        weighted.columns
    )


def test_beamer_figure_references_are_packaged_and_no_red_caveats() -> None:
    tex = Path("presentations/estado_actual/main.tex").read_text(encoding="utf-8")
    referenced = set()
    for part in tex.split("\\includegraphics")[1:]:
        name = part.split("{", 1)[1].split("}", 1)[0]
        referenced.add(Path(name).name)
    packaged = set(REQUIRED_FIGURES) | set(GENERATED_FIGURES)

    assert referenced <= packaged
    assert "\\textcolor{red}" not in tex
