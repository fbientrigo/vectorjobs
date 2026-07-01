import json
from pathlib import Path

import numpy as np
import pandas as pd

from jobsrec.trends.cluster_movement import (
    CLUSTER_TO_GLOBAL_COLUMNS,
    DESCRIPTOR_COLUMNS,
    PAIRWISE_COLUMNS,
    SELF_DRIFT_COLUMNS,
    build_top_titles_skills_by_cluster_week,
    compute_cluster_self_drift,
    compute_cluster_to_global_distance,
    compute_mean_pairwise_cluster_distance,
    run_cluster_movement,
    weekly_cluster_centroids,
)


def _centroids() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "week": ["2024-01-01", "2024-01-01", "2024-01-01"],
            "cluster_id": [0, 1, 2],
            "n_jobs": [10, 20, 30],
            "centroid": [
                np.array([1.0, 0.0]),
                np.array([0.0, 1.0]),
                np.array([1.0, 1.0]),
            ],
        }
    )


def test_cosine_distance_to_global_centroid_is_computed_correctly() -> None:
    centroids = pd.DataFrame(
        {
            "week": ["2024-01-01", "2024-01-01"],
            "cluster_id": [0, 1],
            "n_jobs": [1, 1],
            "centroid": [np.array([1.0, 0.0]), np.array([0.0, 1.0])],
        }
    )

    out = compute_cluster_to_global_distance(centroids)

    np.testing.assert_allclose(out["cluster_to_global_distance"], [1 - 1 / np.sqrt(2)] * 2)


def test_mean_pairwise_distance_is_computed_correctly() -> None:
    out = compute_mean_pairwise_cluster_distance(_centroids())

    expected = np.mean([1.0, 1 - 1 / np.sqrt(2), 1 - 1 / np.sqrt(2)])
    assert out.loc[0, "n_cluster_pairs"] == 3
    assert out.loc[0, "mean_pairwise_distance"] == expected


def test_weighted_pairwise_distance_uses_cluster_size_products() -> None:
    out = compute_mean_pairwise_cluster_distance(_centroids())

    distances = np.array([1.0, 1 - 1 / np.sqrt(2), 1 - 1 / np.sqrt(2)])
    weights = np.array([10 * 20, 10 * 30, 20 * 30])
    assert out.loc[0, "weighted_mean_pairwise_distance"] == np.average(distances, weights=weights)


def test_self_drift_compares_previous_available_week_and_handles_missing_weeks() -> None:
    centroids = pd.DataFrame(
        {
            "week": ["2024-01-01", "2024-01-15", "2024-01-01", "2024-01-08"],
            "cluster_id": [0, 0, 1, 1],
            "n_jobs": [10, 12, 100, 90],
            "centroid": [
                np.array([1.0, 0.0]),
                np.array([0.0, 1.0]),
                np.array([1.0, 1.0]),
                np.array([1.0, 1.0]),
            ],
        }
    )

    out = compute_cluster_self_drift(centroids)
    row = out[out["cluster_id"].eq(0)].iloc[0]

    assert row["prev_week"] == "2024-01-01"
    assert row["week"] == "2024-01-15"
    assert row["self_drift"] == 1.0


def test_low_support_clusters_receive_low_confidence() -> None:
    out = compute_cluster_to_global_distance(_centroids())

    assert out.loc[out["cluster_id"].eq(0), "confidence_label"].iloc[0] == "low"


def test_descriptor_builder_joins_jobs_and_candidates_by_string_job_id_and_parses_skills() -> None:
    assignments = pd.DataFrame(
        {
            "job_id": [1, 2, 3],
            "cluster_id": [0, 0, 1],
            "time_bin": ["2024-01-01", "2024-01-01", "2024-01-01"],
        }
    )
    jobs = pd.DataFrame(
        {
            "job_id": ["1", "2", "3"],
            "title": ["Analista", "Analista", "Vendedor"],
            "company_industry": ["Retail", "Retail", "Ventas"],
        }
    )
    candidates = pd.DataFrame(
        {
            "job_id": ["1", "1", "2", "3"],
            "candidate_source": ["paragraph", "paragraph", "paragraph", "title"],
            "section_name": ["requisitos", "habilidades", "conocimientos", "requisitos"],
            "skills_normalized": [json.dumps(["Excel", "SAP"]), json.dumps([]), json.dumps(["Excel"]), json.dumps(["CRM"])],
        }
    )

    out = build_top_titles_skills_by_cluster_week(assignments, jobs, candidates)
    skills = json.loads(out.loc[out["cluster_id"].eq(0), "top_skills_json"].iloc[0])
    titles = json.loads(out.loc[out["cluster_id"].eq(0), "top_titles_json"].iloc[0])

    assert skills[0] == {"value": "Excel", "count": 2}
    assert {"value": "SAP", "count": 1} in skills
    assert all(item["value"] for item in skills)
    assert titles[0] == {"value": "Analista", "count": 2}


def test_weekly_cluster_centroids_schema_and_values() -> None:
    assignments = pd.DataFrame(
        {
            "job_id": ["1", "2", "3"],
            "cluster_id": [0, 0, 1],
            "time_bin": ["2024-01-01", "2024-01-01", "2024-01-08"],
        }
    )
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]])

    out = weekly_cluster_centroids(assignments, embeddings)

    row = out[out["cluster_id"].eq(0)].iloc[0]
    np.testing.assert_allclose(row["centroid"], [0.5, 0.5])
    assert row["n_jobs"] == 2


def test_output_schemas_contain_expected_columns(tmp_path: Path) -> None:
    clusters_dir = tmp_path / "clusters"
    clusters_dir.mkdir()
    assignments = pd.DataFrame(
        {
            "job_id": ["1", "2", "3", "4"],
            "cluster_id": [0, 1, 0, 1],
            "time_bin": ["2024-01-01", "2024-01-01", "2024-01-08", "2024-01-08"],
            "posted_at": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-08", "2024-01-08"]),
        }
    )
    jobs = pd.DataFrame(
        {
            "job_id": ["1", "2", "3", "4"],
            "title": ["Data Analyst", "Nurse", "Data Engineer", "Care Nurse"],
            "job_card_text": ["python sql", "patient care", "python data", "clinic care"],
            "first_seen_at": ["2024-01-01", "2024-01-01", "2024-01-08", "2024-01-08"],
            "company_industry": ["Tech", "Health", "Tech", "Health"],
        }
    )
    candidates = pd.DataFrame(
        {
            "job_id": ["1", "2", "3", "4"],
            "candidate_source": ["paragraph"] * 4,
            "section_name": ["requisitos"] * 4,
            "skills_normalized": [json.dumps(["Python"]), json.dumps([]), json.dumps(["SQL"]), json.dumps(["Care"])],
        }
    )
    assignments.to_parquet(clusters_dir / "cluster_assignments.parquet", index=False)
    (clusters_dir / "manifest.json").write_text(json.dumps({"random_seed": 42, "schema_mapping": {"time_column": "first_seen_at"}}))
    jobs_path = tmp_path / "jobs.parquet"
    candidates_path = tmp_path / "candidates.parquet"
    jobs.to_parquet(jobs_path, index=False)
    candidates.to_parquet(candidates_path, index=False)

    run_cluster_movement(clusters_dir, jobs_path, candidates_path, clusters_dir)

    assert list(pd.read_parquet(clusters_dir / "cluster_to_global_distance_by_week.parquet").columns) == CLUSTER_TO_GLOBAL_COLUMNS
    assert list(pd.read_parquet(clusters_dir / "mean_pairwise_cluster_distance_by_week.parquet").columns) == PAIRWISE_COLUMNS
    assert list(pd.read_parquet(clusters_dir / "cluster_self_drift_by_week.parquet").columns) == SELF_DRIFT_COLUMNS
    assert list(pd.read_parquet(clusters_dir / "top_titles_skills_by_cluster_week.parquet").columns) == DESCRIPTOR_COLUMNS
    assert (clusters_dir / "cluster_movement_interpretation.md").exists()
