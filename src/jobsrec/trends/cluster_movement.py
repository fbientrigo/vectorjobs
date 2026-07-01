"""Interpretable weekly movement metrics for temporal clusters."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jobsrec.trends.temporal_clusters import (
    build_temporal_cluster_embeddings,
    detect_temporal_cluster_schema,
    _join_text_columns,
)


DISTANCE_METRIC = "cosine"
CLUSTER_TO_GLOBAL_COLUMNS = [
    "week",
    "cluster_id",
    "n_jobs",
    "cluster_to_global_distance",
    "distance_metric",
    "confidence_label",
]
PAIRWISE_COLUMNS = [
    "week",
    "n_active_clusters",
    "n_cluster_pairs",
    "mean_pairwise_distance",
    "median_pairwise_distance",
    "min_pairwise_distance",
    "max_pairwise_distance",
    "weighted_mean_pairwise_distance",
    "distance_metric",
    "confidence_label",
]
SELF_DRIFT_COLUMNS = [
    "week",
    "prev_week",
    "cluster_id",
    "n_jobs",
    "prev_n_jobs",
    "self_drift",
    "delta_n_jobs",
    "distance_metric",
    "confidence_label",
]
DESCRIPTOR_COLUMNS = [
    "week",
    "cluster_id",
    "n_jobs",
    "top_titles_json",
    "top_skills_json",
    "top_industries_json",
    "descriptor_text",
]


@dataclass(frozen=True)
class ClusterMovementResult:
    output_dir: Path
    cluster_to_global_path: Path
    pairwise_path: Path
    self_drift_path: Path
    descriptors_path: Path
    report_path: Path
    generated_files: list[str]


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return float("nan")
    return float(1.0 - np.dot(a, b) / (norm_a * norm_b))


def cluster_confidence(n_jobs: int) -> str:
    if n_jobs < 30:
        return "low"
    if n_jobs < 100:
        return "medium"
    return "high"


def pairwise_confidence(n_active_clusters: int, cluster_sizes: list[int]) -> str:
    if n_active_clusters < 4:
        return "low"
    low_count = sum(size < 30 for size in cluster_sizes)
    if low_count >= math.ceil(n_active_clusters / 2):
        return "low"
    return "high" if all(size >= 100 for size in cluster_sizes) else "medium"


def weekly_cluster_centroids(assignments: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    if len(assignments) != len(embeddings):
        raise ValueError("assignments and embeddings must have the same row count.")
    if "time_bin" not in assignments.columns:
        raise ValueError("cluster assignments must contain time_bin.")

    rows: list[dict[str, Any]] = []
    valid = assignments[assignments["time_bin"].notna()].copy()
    valid["_embedding_index"] = np.arange(len(assignments))[valid.index]
    for (week, cluster_id), group in valid.groupby(["time_bin", "cluster_id"], sort=True):
        idx = group["_embedding_index"].to_numpy(dtype=int)
        rows.append(
            {
                "week": str(week),
                "cluster_id": int(cluster_id),
                "n_jobs": int(len(group)),
                "centroid": embeddings[idx].mean(axis=0),
            }
        )
    return pd.DataFrame(rows)


def compute_cluster_to_global_distance(centroids: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for week, group in centroids.groupby("week", sort=True):
        weights = group["n_jobs"].to_numpy(dtype=float)
        vectors = np.vstack(group["centroid"].to_numpy())
        global_centroid = np.average(vectors, axis=0, weights=weights)
        for _, row in group.iterrows():
            n_jobs = int(row["n_jobs"])
            rows.append(
                {
                    "week": str(week),
                    "cluster_id": int(row["cluster_id"]),
                    "n_jobs": n_jobs,
                    "cluster_to_global_distance": cosine_distance(row["centroid"], global_centroid),
                    "distance_metric": DISTANCE_METRIC,
                    "confidence_label": cluster_confidence(n_jobs),
                }
            )
    return pd.DataFrame(rows, columns=CLUSTER_TO_GLOBAL_COLUMNS)


def compute_mean_pairwise_cluster_distance(centroids: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for week, group in centroids.groupby("week", sort=True):
        distances: list[float] = []
        weights: list[float] = []
        records = group.to_dict(orient="records")
        for i, left in enumerate(records):
            for right in records[i + 1 :]:
                distance = cosine_distance(left["centroid"], right["centroid"])
                distances.append(distance)
                weights.append(float(left["n_jobs"]) * float(right["n_jobs"]))
        n_active = int(len(records))
        sizes = [int(row["n_jobs"]) for row in records]
        rows.append(
            {
                "week": str(week),
                "n_active_clusters": n_active,
                "n_cluster_pairs": int(len(distances)),
                "mean_pairwise_distance": float(np.mean(distances)) if distances else np.nan,
                "median_pairwise_distance": float(np.median(distances)) if distances else np.nan,
                "min_pairwise_distance": float(np.min(distances)) if distances else np.nan,
                "max_pairwise_distance": float(np.max(distances)) if distances else np.nan,
                "weighted_mean_pairwise_distance": float(np.average(distances, weights=weights)) if distances else np.nan,
                "distance_metric": DISTANCE_METRIC,
                "confidence_label": pairwise_confidence(n_active, sizes),
            }
        )
    return pd.DataFrame(rows, columns=PAIRWISE_COLUMNS)


def compute_cluster_self_drift(centroids: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cluster_id, group in centroids.sort_values("week").groupby("cluster_id", sort=True):
        previous: dict[str, Any] | None = None
        for row in group.to_dict(orient="records"):
            if previous is not None:
                n_jobs = int(row["n_jobs"])
                prev_n_jobs = int(previous["n_jobs"])
                rows.append(
                    {
                        "week": str(row["week"]),
                        "prev_week": str(previous["week"]),
                        "cluster_id": int(cluster_id),
                        "n_jobs": n_jobs,
                        "prev_n_jobs": prev_n_jobs,
                        "self_drift": cosine_distance(row["centroid"], previous["centroid"]),
                        "delta_n_jobs": int(n_jobs - prev_n_jobs),
                        "distance_metric": DISTANCE_METRIC,
                        "confidence_label": cluster_confidence(min(n_jobs, prev_n_jobs)),
                    }
                )
            previous = row
    return pd.DataFrame(rows, columns=SELF_DRIFT_COLUMNS)


def _top_json(counter: Counter[str], n: int = 10) -> str:
    values = [{"value": value, "count": int(count)} for value, count in counter.most_common(n) if value]
    return json.dumps(values, ensure_ascii=False)


def _parse_skills(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, list):
        return []
    return [str(item).strip() for item in parsed if str(item).strip()]


def build_top_titles_skills_by_cluster_week(
    assignments: pd.DataFrame,
    jobs: pd.DataFrame,
    candidates: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    assignments = assignments.copy()
    jobs = jobs.copy()
    assignments["job_id"] = assignments["job_id"].astype(str)
    jobs["job_id"] = jobs["job_id"].astype(str)

    title_col = "title_clean" if "title_clean" in jobs.columns else "title"
    industry_col = "company_industry" if "company_industry" in jobs.columns else None
    merged = assignments.merge(jobs, on="job_id", how="left", suffixes=("", "_job"))

    skill_by_job: dict[str, list[str]] = {}
    if not candidates.empty:
        cand = candidates.copy()
        cand["job_id"] = cand["job_id"].astype(str)
        cand = cand[
            cand["candidate_source"].isin(["li", "paragraph"])
            & cand["section_name"].isin(["requisitos", "habilidades", "conocimientos"])
        ]
        for job_id, group in cand.groupby("job_id", sort=False):
            skills: list[str] = []
            for value in group["skills_normalized"]:
                skills.extend(_parse_skills(value))
            skill_by_job[str(job_id)] = skills

    rows: list[dict[str, Any]] = []
    for (week, cluster_id), group in merged.groupby(["time_bin", "cluster_id"], sort=True):
        title_counter = Counter(str(value).strip() for value in group.get(title_col, pd.Series(dtype=str)).dropna() if str(value).strip())
        industry_counter = Counter()
        if industry_col:
            industry_counter.update(str(value).strip() for value in group[industry_col].dropna() if str(value).strip())
        skill_counter: Counter[str] = Counter()
        for job_id in group["job_id"].astype(str):
            skill_counter.update(skill_by_job.get(job_id, []))

        top_titles = title_counter.most_common(3)
        top_skills = skill_counter.most_common(5)
        top_industries = industry_counter.most_common(3)
        parts = [f"C{int(cluster_id):02d}", str(week), f"{len(group)} jobs"]
        if top_titles:
            parts.append("titles: " + ", ".join(value for value, _ in top_titles))
        if top_skills:
            parts.append("skills: " + ", ".join(value for value, _ in top_skills))
        if top_industries:
            parts.append("industries: " + ", ".join(value for value, _ in top_industries))
        rows.append(
            {
                "week": str(week),
                "cluster_id": int(cluster_id),
                "n_jobs": int(len(group)),
                "top_titles_json": _top_json(title_counter, top_n),
                "top_skills_json": _top_json(skill_counter, top_n),
                "top_industries_json": _top_json(industry_counter, top_n),
                "descriptor_text": " | ".join(parts),
            }
        )
    return pd.DataFrame(rows, columns=DESCRIPTOR_COLUMNS)


def _trend_delta(df: pd.DataFrame, column: str) -> float:
    valid = df.dropna(subset=[column]).sort_values("week")
    if len(valid) < 2:
        return float("nan")
    return float(valid[column].iloc[-1] - valid[column].iloc[0])


def _verdict(pairwise: pd.DataFrame, to_global: pd.DataFrame) -> str:
    pair_delta = _trend_delta(pairwise, "mean_pairwise_distance")
    global_weekly = (
        to_global.assign(weight=lambda df: df["n_jobs"].astype(float))
        .groupby("week")
        .apply(lambda g: np.average(g["cluster_to_global_distance"], weights=g["weight"]), include_groups=False)
        .reset_index(name="weighted_global_distance")
    )
    global_delta = _trend_delta(global_weekly, "weighted_global_distance")
    if math.isnan(pair_delta):
        return "insufficient evidence"
    baseline = float(pairwise["mean_pairwise_distance"].dropna().iloc[0])
    threshold = max(abs(baseline) * 0.05, 0.001)
    if abs(pair_delta) <= threshold:
        return "stable"
    if pair_delta < 0 and (math.isnan(global_delta) or global_delta <= threshold):
        return "convergence"
    if pair_delta > 0 and (math.isnan(global_delta) or global_delta >= -threshold):
        return "divergence"
    return "mixed"


def write_interpretation_report(path: Path, pairwise: pd.DataFrame, to_global: pd.DataFrame, self_drift: pd.DataFrame) -> None:
    verdict = _verdict(pairwise, to_global)
    pair_delta = _trend_delta(pairwise, "mean_pairwise_distance")
    global_weekly = (
        to_global.assign(weight=lambda df: df["n_jobs"].astype(float))
        .groupby("week")
        .apply(lambda g: np.average(g["cluster_to_global_distance"], weights=g["weight"]), include_groups=False)
        .reset_index(name="weighted_global_distance")
    )
    global_delta = _trend_delta(global_weekly, "weighted_global_distance")
    high_drift = self_drift.sort_values("self_drift", ascending=False).head(5)
    stable = self_drift.sort_values("self_drift", ascending=True).head(5)

    lines = [
        "# Cluster Movement Interpretation",
        "",
        f"## Summary verdict: {verdict}",
        "",
        "## Evidence",
        "",
        f"- Mean pairwise cluster distance change: {pair_delta:.4f}" if not math.isnan(pair_delta) else "- Mean pairwise cluster distance change: insufficient data",
        f"- Weighted distance to weekly global center change: {global_delta:.4f}" if not math.isnan(global_delta) else "- Weighted distance to weekly global center change: insufficient data",
        "- Highest self-drift clusters: "
        + (", ".join(f"C{int(r.cluster_id):02d} {r.prev_week}->{r.week} ({r.self_drift:.4f})" for r in high_drift.itertuples()) or "insufficient data"),
        "- Most stable cluster identities: "
        + (", ".join(f"C{int(r.cluster_id):02d} {r.prev_week}->{r.week} ({r.self_drift:.4f})" for r in stable.itertuples()) or "insufficient data"),
        "",
        "## Caveats",
        "",
        "- Limited temporal coverage can make trends fragile.",
        "- Weekly bins may be noisy, especially for low-support clusters.",
        "- Movement reflects composition and language of observed job ads, not causal labor-market evolution.",
        "- Clustering is representation-dependent.",
        "- Regex skills cover named hard skills and miss many domain skills.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_line_plot(df: pd.DataFrame, x_col: str, y_col: str, path: Path, title: str, ylabel: str, group_col: str | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    if df.empty:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax.axis("off")
    elif group_col:
        for key, group in df.groupby(group_col, sort=True):
            ax.plot(group[x_col].astype(str), group[y_col], linewidth=1, alpha=0.7, label=f"C{int(key):02d}")
        ax.legend(loc="best", fontsize=7, ncol=2)
    else:
        ax.plot(df[x_col].astype(str), df[y_col], marker="o", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Week")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _read_manifest(clusters_dir: Path) -> dict[str, Any]:
    path = clusters_dir / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def run_cluster_movement(
    clusters_dir: Path,
    jobs_path: Path,
    candidates_path: Path,
    output_dir: Path,
) -> ClusterMovementResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _read_manifest(clusters_dir)
    assignments = pd.read_parquet(clusters_dir / "cluster_assignments.parquet")
    jobs = pd.read_parquet(jobs_path)
    candidates = pd.read_parquet(candidates_path) if candidates_path.exists() else pd.DataFrame()

    if "job_id" not in assignments.columns:
        raise ValueError("cluster_assignments.parquet must contain job_id.")
    assignments = assignments.copy()
    assignments["job_id"] = assignments["job_id"].astype(str)
    jobs = jobs.copy()
    jobs["job_id"] = jobs["job_id"].astype(str)
    joined = assignments.merge(jobs, on="job_id", how="left", suffixes=("", "_job"), indicator=True)
    if joined["_merge"].ne("both").any():
        raise ValueError("cluster assignments could not be joined to jobs by string job_id.")
    joined = joined.drop(columns=["_merge"])

    schema = detect_temporal_cluster_schema(jobs, time_column=manifest.get("schema_mapping", {}).get("time_column"))
    joined["_analysis_text"] = _join_text_columns(joined, schema["text_columns"])
    embeddings, _, _, _, _ = build_temporal_cluster_embeddings(
        joined["_analysis_text"],
        embedding="tfidf_svd",
        random_state=int(manifest.get("random_seed", 42)),
    )

    centroids = weekly_cluster_centroids(joined[["job_id", "cluster_id", "time_bin"]], embeddings)
    to_global = compute_cluster_to_global_distance(centroids)
    pairwise = compute_mean_pairwise_cluster_distance(centroids)
    self_drift = compute_cluster_self_drift(centroids)
    descriptors = build_top_titles_skills_by_cluster_week(assignments, jobs, candidates)

    cluster_to_global_path = output_dir / "cluster_to_global_distance_by_week.parquet"
    pairwise_path = output_dir / "mean_pairwise_cluster_distance_by_week.parquet"
    self_drift_path = output_dir / "cluster_self_drift_by_week.parquet"
    descriptors_path = output_dir / "top_titles_skills_by_cluster_week.parquet"
    report_path = output_dir / "cluster_movement_interpretation.md"
    to_global.to_parquet(cluster_to_global_path, index=False)
    pairwise.to_parquet(pairwise_path, index=False)
    self_drift.to_parquet(self_drift_path, index=False)
    descriptors.to_parquet(descriptors_path, index=False)
    write_interpretation_report(report_path, pairwise, to_global, self_drift)

    figure_paths = [
        output_dir / "cluster_to_global_distance_by_week.png",
        output_dir / "mean_pairwise_cluster_distance_by_week.png",
        output_dir / "cluster_self_drift_by_week.png",
    ]
    _write_line_plot(to_global, "week", "cluster_to_global_distance", figure_paths[0], "Cluster Distance to Weekly Global Center", "Cosine distance", "cluster_id")
    _write_line_plot(pairwise, "week", "mean_pairwise_distance", figure_paths[1], "Mean Pairwise Cluster Distance by Week", "Cosine distance")
    _write_line_plot(self_drift, "week", "self_drift", figure_paths[2], "Cluster Self Drift by Week", "Cosine distance", "cluster_id")

    generated = [cluster_to_global_path, pairwise_path, self_drift_path, descriptors_path, report_path, *figure_paths]
    return ClusterMovementResult(
        output_dir=output_dir,
        cluster_to_global_path=cluster_to_global_path,
        pairwise_path=pairwise_path,
        self_drift_path=self_drift_path,
        descriptors_path=descriptors_path,
        report_path=report_path,
        generated_files=[str(path) for path in generated],
    )
