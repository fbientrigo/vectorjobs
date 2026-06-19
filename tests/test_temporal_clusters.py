import json
from pathlib import Path

import numpy as np
import pandas as pd
from click.testing import CliRunner

from jobsrec.cli import main
from jobsrec.trends.temporal_clusters import (
    build_cluster_labels,
    build_temporal_cluster_embeddings,
    compute_decay_inputs,
    compute_temporal_cluster_metrics,
    detect_temporal_cluster_schema,
    fit_exponential_decay_by_cluster,
    fit_fixed_clusters,
    prepare_cluster_trajectory_change_data,
)


def _cluster_jobs(include_closure: bool = True) -> pd.DataFrame:
    rows = []
    for i in range(12):
        is_data = i % 2 == 0
        month = 1 if i < 6 else 2
        rows.append(
            {
                "job_id": i + 1,
                "title": "Data Analyst" if is_data else "Nurse Practitioner",
                "description": (
                    "python sql warehouse analytics dashboards"
                    if is_data
                    else "patient clinic nursing care medication"
                ),
                "job_card_text": (
                    f"python sql data analytics warehouse {i}"
                    if is_data
                    else f"patient clinic nursing care medication {i}"
                ),
                "skills_text": "python; sql; analytics" if is_data else "nursing; patient care",
                "listed_time": f"2024-0{month}-{(i % 6) + 1:02d}",
                "closed_time": f"2024-0{month}-{(i % 6) + 6:02d}" if include_closure else None,
                "min_salary": 50000.0 if i % 3 != 0 else np.nan,
                "max_salary": 70000.0 if i % 3 != 0 else np.nan,
                "med_salary": np.nan,
                "normalized_salary": np.nan,
                "pay_period": "YEARLY",
                "currency": "USD",
            }
        )
    df = pd.DataFrame(rows)
    if not include_closure:
        df = df.drop(columns=["closed_time"])
    return df


def test_fixed_cluster_assignment_is_stable_with_random_state() -> None:
    embeddings = np.array(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [1.0, 0.1], [0.1, 1.0]],
        dtype=np.float32,
    )

    first_ids, first_centroids, _ = fit_fixed_clusters(embeddings, k=2, random_state=42)
    second_ids, second_centroids, _ = fit_fixed_clusters(embeddings, k=2, random_state=42)

    assert first_ids.tolist() == second_ids.tolist()
    np.testing.assert_allclose(first_centroids, second_centroids)


def test_temporal_aggregation_produces_expected_bins_shares_and_salary_coverage() -> None:
    df = _cluster_jobs(include_closure=False).head(4).copy()
    df["_embedding_index"] = np.arange(len(df))
    df["_posted_at"] = pd.to_datetime(df["listed_time"])
    df["time_bin"] = df["_posted_at"].dt.to_period("M").astype("string")
    df["cluster_id"] = [0, 0, 1, 1]
    embeddings = np.eye(4, dtype=np.float32)
    centroids = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    labels = pd.DataFrame(
        {
            "cluster_id": [0, 1],
            "cluster_label": ["C00 | data", "C01 | care"],
            "n_jobs": [2, 2],
        }
    )

    metrics = compute_temporal_cluster_metrics(df, embeddings, centroids, labels, skill_column="skills_text")

    assert metrics["time_bin"].tolist() == ["2024-01", "2024-01"]
    assert metrics["share_jobs"].tolist() == [0.5, 0.5]
    assert metrics.loc[metrics["cluster_id"] == 0, "salary_coverage"].iloc[0] == 0.5
    assert metrics.loc[metrics["cluster_id"] == 1, "n_with_skills"].iloc[0] == 2


def test_survival_exponential_fit_returns_expected_lambda() -> None:
    durations = pd.DataFrame(
        {
            "cluster_id": [0, 0, 0],
            "duration_days": [10.0, 20.0, 30.0],
            "event_observed": [True, False, True],
        }
    )
    labels = pd.DataFrame({"cluster_id": [0], "cluster_label": ["C00 | data"]})

    summary = fit_exponential_decay_by_cluster(durations, labels, min_rows=1, min_events=1)

    assert summary.loc[0, "lambda"] == 2.0 / 60.0
    assert summary.loc[0, "n_events"] == 2
    assert summary.loc[0, "n_censored"] == 1


def test_no_closure_or_last_seen_columns_skip_survival() -> None:
    df = _cluster_jobs(include_closure=False)
    schema = detect_temporal_cluster_schema(df)
    df["_posted_at"] = pd.to_datetime(df[schema["time_column"]])
    df["cluster_id"] = 0

    durations, reason = compute_decay_inputs(df, schema)

    assert durations.empty
    assert "No closure" in reason


def test_cluster_labels_are_non_empty() -> None:
    df = _cluster_jobs(include_closure=False)
    embeddings, vectorizer, _, tfidf, _ = build_temporal_cluster_embeddings(
        df["job_card_text"],
        embedding="tfidf_svd",
        random_state=42,
    )
    cluster_ids, _, _ = fit_fixed_clusters(embeddings, k=2, random_state=42)

    labels = build_cluster_labels(df, cluster_ids, tfidf, vectorizer, skill_column="skills_text")

    assert len(labels) == 2
    assert labels["cluster_label"].str.len().gt(0).all()
    assert labels["cluster_label"].str.startswith("C").all()


def test_temporal_clusters_cli_writes_required_outputs_and_skip_artifact(tmp_path: Path) -> None:
    silver_path = tmp_path / "jobs.parquet"
    outdir = tmp_path / "temporal_clusters"
    _cluster_jobs(include_closure=False).to_parquet(silver_path, index=False)

    result = CliRunner().invoke(
        main,
        [
            "temporal-clusters",
            "--input",
            str(silver_path),
            "--outdir",
            str(outdir),
            "--bin",
            "M",
            "--k",
            "3",
            "--embedding",
            "tfidf_svd",
            "--max-rows",
            "12",
            "--random-state",
            "42",
        ],
    )

    assert result.exit_code == 0, result.output
    manifest = json.loads((outdir / "manifest.json").read_text())
    metrics = pd.read_parquet(outdir / "cluster_time_metrics.parquet")
    assert manifest["selected_row_count"] == 12
    assert manifest["decay_available"] is False
    assert not metrics.empty
    assert (outdir / "cluster_bubble_timeline.png").exists()
    assert (outdir / "cluster_share_timeseries.png").exists()
    assert (outdir / "cluster_semantic_trajectory.png").exists()
    assert (outdir / "decay_not_available.md").exists()
    assert (outdir / "report.md").exists()
    assert (outdir / "cluster_semantic_trajectory.png").stat().st_size > 0


def test_cluster_trajectory_change_data_limits_to_top_n_clusters() -> None:
    df = pd.DataFrame(
        {
            "cluster_id": [0, 0, 1, 1, 1, 2, 2, 3],
            "time_bin": ["2024-03", "2024-04", "2024-03", "2024-04", "2024-04", "2024-03", "2024-04", "2024-04"],
            "_embedding_index": list(range(8)),
        }
    )
    embeddings = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [0.2, 2.2],
            [2.0, 0.0],
            [2.5, 0.5],
            [3.0, 3.0],
        ],
        dtype=np.float32,
    )
    labels = pd.DataFrame(
        {
            "cluster_id": [0, 1, 2, 3],
            "cluster_label": ["C00 | data", "C01 | care", "C02 | sales", "C03 | other"],
        }
    )

    change = prepare_cluster_trajectory_change_data(df, embeddings, labels, random_state=42, top_n=2)

    assert len(change) == 2
    assert set(change["cluster_id"]) == {0, 1}
    assert change["projected_distance"].notna().all()
