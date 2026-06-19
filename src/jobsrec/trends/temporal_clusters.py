"""Fixed temporal cluster analytics for job postings."""

from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from jobsrec.trends.temporal import compute_annual_salary


TEXT_CANDIDATES = ["job_card_text", "description", "title"]
SKILL_CANDIDATES = ["skills_text", "skills_desc"]
TIME_CANDIDATES = ["posted_at", "posting_date", "created_at", "listed_time", "original_listed_time"]
CLOSURE_CANDIDATES = ["closed_at", "expired_at", "taken_at", "closed_time", "expiry"]
LAST_SEEN_CANDIDATES = ["last_seen"]
SALARY_COLUMNS = ["normalized_salary", "med_salary", "min_salary", "max_salary"]
METRIC_COLUMNS = [
    "time_bin",
    "cluster_id",
    "cluster_label",
    "n_jobs",
    "share_jobs",
    "salary_median",
    "salary_mean",
    "salary_coverage",
    "top_skills",
    "centroid_drift_from_global",
    "centroid_drift_from_previous_bin",
    "n_with_salary",
    "n_with_skills",
]
FOCUS_CLUSTER_DOMAINS = [
    {
        "key": "health",
        "label": "Health",
        "color": "#d62728",
        "keywords": ["patient", "nursing", "nurse", "medical", "clinic", "health care", "healthcare", "health"],
    },
    {
        "key": "technology",
        "label": "Tech",
        "color": "#1f77b4",
        "keywords": ["software", "developer", "engineer", "engineering", "information technology", "technology", "data", "python", "sql"],
    },
    {
        "key": "sales",
        "label": "Sales",
        "color": "#2ca02c",
        "keywords": ["sales", "sales", "sales", "business", "business", "business development", "customer service", "account"],
    },
    {
        "key": "construction",
        "label": "Construction",
        "color": "#ff7f0e",
        "keywords": ["construction", "project management", "project", "building", "site", "contractor"],
    },
    {
        "key": "education",
        "label": "Education",
        "color": "#9467bd",
        "keywords": ["education", "students", "student", "teacher", "teaching", "school", "training"],
    },
]


@dataclass(frozen=True)
class TemporalClusterResult:
    """Paths and manifest for a temporal cluster run."""

    output_dir: Path
    manifest_path: Path
    report_path: Path
    metrics_path: Path
    generated_files: list[str]
    manifest: dict[str, Any]


def _first_present(columns: pd.Index, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _parse_datetime(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().any():
        parsed_years = parsed.dropna().dt.year
        parsed_suspicious = parsed_years.empty or parsed_years.median() < 1990
        if parsed_suspicious:
            for unit in ("ms", "s"):
                reparsed = pd.to_datetime(numeric, errors="coerce", unit=unit)
                years = reparsed.dropna().dt.year
                if not years.empty and 1990 <= years.median() <= 2100:
                    return reparsed
    return parsed


def _safe_strings(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str)


def _join_text_columns(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    present = [column for column in columns if column in df.columns]
    if not present:
        raise ValueError(f"Input data must contain at least one text column from {TEXT_CANDIDATES}.")
    text = _safe_strings(df[present[0]])
    for column in present[1:]:
        text = text.str.cat(_safe_strings(df[column]), sep=" ")
    return text.str.strip()


def _split_skills(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    parts = text.replace("|", ";").replace(",", ";").split(";")
    return [" ".join(part.lower().split()) for part in parts if part.strip()]


def detect_temporal_cluster_schema(df: pd.DataFrame, time_column: str | None = None) -> dict[str, Any]:
    """Detect useful columns without assuming one exact silver schema."""
    text_columns = [column for column in TEXT_CANDIDATES if column in df.columns]
    if time_column is not None and time_column not in df.columns:
        raise ValueError(f"Requested time column not found: {time_column}")
    detected_time_column = time_column or _first_present(df.columns, TIME_CANDIDATES)
    skill_column = _first_present(df.columns, SKILL_CANDIDATES)
    closure_column = _first_present(df.columns, CLOSURE_CANDIDATES)
    last_seen_column = _first_present(df.columns, LAST_SEEN_CANDIDATES)
    salary_columns = [column for column in SALARY_COLUMNS + ["currency", "pay_period"] if column in df.columns]
    if not text_columns:
        raise ValueError(f"No supported text columns found. Tried: {TEXT_CANDIDATES}.")
    if detected_time_column is None:
        raise ValueError(f"No supported posting time column found. Tried: {TIME_CANDIDATES}.")
    return {
        "text_columns": text_columns,
        "time_column": detected_time_column,
        "skill_column": skill_column,
        "closure_column": closure_column,
        "last_seen_column": last_seen_column,
        "salary_columns": salary_columns,
        "salary_available": any(column in df.columns for column in SALARY_COLUMNS),
        "decay_available": closure_column is not None or last_seen_column is not None,
    }


def _select_rows(df: pd.DataFrame, max_rows: int | None, random_state: int) -> pd.DataFrame:
    if max_rows is None or max_rows <= 0 or len(df) <= max_rows:
        return df.copy().reset_index(drop=True)
    return df.sample(n=max_rows, random_state=random_state).reset_index(drop=True)


def _time_bins(parsed: pd.Series, bin_size: str) -> pd.Series:
    if bin_size.upper() == "W":
        return parsed.dt.to_period("W").astype("string")
    if bin_size.upper() == "D":
        return parsed.dt.to_period("D").astype("string")
    return parsed.dt.to_period("M").astype("string")


def _build_tfidf_svd_embeddings(
    texts: pd.Series,
    random_state: int,
    max_features: int = 20000,
) -> tuple[np.ndarray, TfidfVectorizer, Any | None, Any, dict[str, Any]]:
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=1,
        max_df=0.95,
        ngram_range=(1, 2),
        stop_words="english",
    )
    tfidf = vectorizer.fit_transform(_safe_strings(texts))
    n_components = min(50, max(0, min(tfidf.shape) - 1))
    svd = None
    if n_components >= 2:
        svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        embeddings = svd.fit_transform(tfidf)
        method = "tfidf_svd"
    else:
        embeddings = tfidf.toarray()
        method = "tfidf"
    embeddings = normalize(np.asarray(embeddings, dtype=np.float32), copy=False)
    metadata = {
        "embedding_method": method,
        "tfidf_vocabulary_size": int(len(vectorizer.vocabulary_)),
        "svd_components": int(n_components) if svd is not None else 0,
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
    }
    return embeddings, vectorizer, svd, tfidf, metadata


def _build_sentence_transformer_embeddings(
    texts: pd.Series,
    model_name: str,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")
    embeddings = model.encode(
        _safe_strings(texts).tolist(),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    return embeddings, {
        "embedding_method": "sentence_transformers",
        "embedding_model": model_name,
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
    }


def build_temporal_cluster_embeddings(
    texts: pd.Series,
    embedding: str,
    random_state: int,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    embedding_batch_size: int = 8,
) -> tuple[np.ndarray, TfidfVectorizer, Any | None, Any, dict[str, Any]]:
    """Build CPU-safe baseline embeddings, with optional graceful dense fallback."""
    if embedding == "sentence_transformers":
        try:
            embeddings, metadata = _build_sentence_transformer_embeddings(
                texts,
                model_name=embedding_model,
                batch_size=embedding_batch_size,
            )
            vectorizer = TfidfVectorizer(max_features=20000, min_df=1, stop_words="english")
            tfidf = vectorizer.fit_transform(_safe_strings(texts))
            metadata["embedding_fallback_reason"] = None
            return embeddings, vectorizer, None, tfidf, metadata
        except Exception as exc:
            embeddings, vectorizer, svd, tfidf, metadata = _build_tfidf_svd_embeddings(texts, random_state)
            metadata["requested_embedding"] = "sentence_transformers"
            metadata["embedding_fallback_reason"] = str(exc)
            return embeddings, vectorizer, svd, tfidf, metadata
    if embedding != "tfidf_svd":
        raise ValueError(f"Unsupported embedding method: {embedding}")
    embeddings, vectorizer, svd, tfidf, metadata = _build_tfidf_svd_embeddings(texts, random_state)
    metadata["requested_embedding"] = embedding
    metadata["embedding_fallback_reason"] = None
    return embeddings, vectorizer, svd, tfidf, metadata


def fit_fixed_clusters(
    embeddings: np.ndarray,
    k: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, MiniBatchKMeans]:
    """Fit one global clustering model and assign all selected rows to fixed clusters."""
    if len(embeddings) == 0:
        raise ValueError("Cannot cluster an empty dataset.")
    n_clusters = max(1, min(int(k), len(embeddings)))
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=min(2048, max(100, len(embeddings))),
        n_init=10,
    )
    cluster_ids = model.fit_predict(embeddings)
    centroids = normalize(model.cluster_centers_.astype(np.float32), copy=False)
    return cluster_ids.astype(int), centroids, model


def _top_terms_for_cluster(tfidf: Any, indices: np.ndarray, vectorizer: TfidfVectorizer, n_terms: int = 3) -> list[str]:
    if len(indices) == 0:
        return []
    feature_names = np.asarray(vectorizer.get_feature_names_out())
    mean_scores = np.asarray(tfidf[indices].mean(axis=0)).ravel()
    if mean_scores.size == 0:
        return []
    order = mean_scores.argsort()[::-1]
    terms: list[str] = []
    for idx in order:
        term = str(feature_names[idx]).strip()
        if term and term not in terms:
            terms.append(term)
        if len(terms) >= n_terms:
            break
    return terms


def _top_skills_for_values(values: pd.Series, n_skills: int = 5) -> str:
    counter: Counter[str] = Counter()
    for value in values:
        counter.update(_split_skills(value))
    return "; ".join(skill for skill, _ in counter.most_common(n_skills))


def build_cluster_labels(
    df: pd.DataFrame,
    cluster_ids: np.ndarray,
    tfidf: Any,
    vectorizer: TfidfVectorizer,
    skill_column: str | None,
    n_terms: int = 3,
) -> pd.DataFrame:
    """Build concise labels from top text terms and optional skills."""
    rows: list[dict[str, Any]] = []
    clusters = sorted(set(int(cluster_id) for cluster_id in cluster_ids))
    for cluster_id in clusters:
        indices = np.flatnonzero(cluster_ids == cluster_id)
        terms = _top_terms_for_cluster(tfidf, indices, vectorizer, n_terms=n_terms)
        if len(terms) < n_terms and skill_column:
            skills = _top_skills_for_values(df.iloc[indices][skill_column], n_skills=n_terms).split("; ")
            terms.extend([skill for skill in skills if skill and skill not in terms])
        terms = terms[:n_terms] or [f"cluster {cluster_id:02d}"]
        top_skills = _top_skills_for_values(df.iloc[indices][skill_column], n_skills=8) if skill_column else ""
        label = f"C{cluster_id:02d} | " + " / ".join(terms)
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_label": label,
                "top_terms": "; ".join(terms),
                "top_skills": top_skills,
                "n_jobs": int(len(indices)),
            }
        )
    return pd.DataFrame(rows).sort_values("cluster_id", ignore_index=True)


def _cluster_keyword_score(text: str, keywords: list[str]) -> int:
    normalized = " ".join(text.lower().replace("/", " ").replace(";", " ").replace("|", " ").split())
    return sum(normalized.count(keyword.lower()) for keyword in keywords)


def _select_focus_clusters(labels: pd.DataFrame, domain_keys: list[str], max_clusters: int) -> pd.DataFrame:
    if labels.empty or max_clusters <= 0:
        return pd.DataFrame(columns=["cluster_id", "cluster_label", "domain", "display_label", "color"])

    domains = [domain for domain in FOCUS_CLUSTER_DOMAINS if domain["key"] in domain_keys]
    selected_rows: list[dict[str, Any]] = []
    selected_cluster_ids: set[int] = set()

    for domain in domains:
        candidates: list[tuple[int, int, int, pd.Series]] = []
        for _, row in labels.iterrows():
            cluster_id = int(row["cluster_id"])
            if cluster_id in selected_cluster_ids:
                continue
            text = f"{row.get('cluster_label', '')} {row.get('top_terms', '')} {row.get('top_skills', '')}"
            score = _cluster_keyword_score(text, list(domain["keywords"]))
            if score > 0:
                candidates.append((score, int(row.get("n_jobs", 0)), -cluster_id, row))
        if not candidates:
            continue
        _, _, _, best = max(candidates, key=lambda item: item[:3])
        cluster_id = int(best["cluster_id"])
        selected_cluster_ids.add(cluster_id)
        selected_rows.append(
            {
                "cluster_id": cluster_id,
                "cluster_label": str(best["cluster_label"]),
                "domain": str(domain["key"]),
                "display_label": f"{domain['label']} (C{cluster_id:02d})",
                "color": str(domain["color"]),
            }
        )
        if len(selected_rows) >= max_clusters:
            break

    if len(selected_rows) < max_clusters:
        for _, row in labels.sort_values("n_jobs", ascending=False).iterrows():
            cluster_id = int(row["cluster_id"])
            if cluster_id in selected_cluster_ids:
                continue
            selected_rows.append(
                {
                    "cluster_id": cluster_id,
                    "cluster_label": str(row["cluster_label"]),
                    "domain": "other",
                    "display_label": f"Other (C{cluster_id:02d})",
                    "color": "#7f7f7f",
                }
            )
            selected_cluster_ids.add(cluster_id)
            if len(selected_rows) >= max_clusters:
                break

    return pd.DataFrame(selected_rows)


def _time_bin_start(values: pd.Series) -> pd.Series:
    starts = values.astype(str).str.split("/", n=1).str[0]
    return pd.to_datetime(starts, errors="coerce")


def _complete_daily_focus_grid(data: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data
    dates = pd.date_range(data["_time"].min(), data["_time"].max(), freq="D")
    grid = pd.MultiIndex.from_product([dates, selected["cluster_id"]], names=["_time", "cluster_id"]).to_frame(index=False)
    complete = grid.merge(data, on=["_time", "cluster_id"], how="left").merge(
        selected[["cluster_id", "display_label", "color", "cluster_label"]],
        on="cluster_id",
        how="left",
        suffixes=("", "_selected"),
    )
    complete["share_jobs"] = complete["share_jobs"].fillna(0.0)
    complete["n_jobs"] = complete["n_jobs"].fillna(0)
    complete["cluster_label"] = complete["cluster_label"].fillna(complete["cluster_label_selected"])
    return complete.drop(columns=[column for column in ["cluster_label_selected"] if column in complete.columns])


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return float("nan")
    return float(1.0 - np.dot(a, b) / (norm_a * norm_b))


def compute_temporal_cluster_metrics(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    centroids: np.ndarray,
    labels: pd.DataFrame,
    skill_column: str | None,
) -> pd.DataFrame:
    """Aggregate fixed clusters through time."""
    if df.empty:
        return pd.DataFrame(columns=METRIC_COLUMNS)
    label_map = labels.set_index("cluster_id")["cluster_label"].to_dict()
    valid = df[df["time_bin"].notna()].copy()
    if valid.empty:
        return pd.DataFrame(columns=METRIC_COLUMNS)

    salary = compute_annual_salary(valid)
    salary = salary.where(_currency_ok(valid))
    valid["_annual_salary"] = salary
    valid["_has_skills"] = (
        valid[skill_column].fillna("").astype(str).str.strip().ne("") if skill_column else False
    )
    total_by_bin = valid.groupby("time_bin").size().to_dict()
    previous_centroids: dict[int, np.ndarray] = {}
    rows: list[dict[str, Any]] = []

    for (time_bin, cluster_id), group in valid.groupby(["time_bin", "cluster_id"], sort=True):
        indices = group["_embedding_index"].to_numpy(dtype=int)
        temporal_centroid = embeddings[indices].mean(axis=0)
        previous = previous_centroids.get(int(cluster_id))
        drift_previous = _cosine_distance(previous, temporal_centroid) if previous is not None else np.nan
        previous_centroids[int(cluster_id)] = temporal_centroid
        cluster_salary = group["_annual_salary"]
        n_jobs = int(len(group))
        n_with_salary = int(cluster_salary.notna().sum())
        n_with_skills = int(group["_has_skills"].sum()) if skill_column else 0
        rows.append(
            {
                "time_bin": str(time_bin),
                "cluster_id": int(cluster_id),
                "cluster_label": label_map.get(int(cluster_id), f"C{int(cluster_id):02d}"),
                "n_jobs": n_jobs,
                "share_jobs": float(n_jobs / total_by_bin[time_bin]) if total_by_bin.get(time_bin) else 0.0,
                "salary_median": float(cluster_salary.median()) if n_with_salary else np.nan,
                "salary_mean": float(cluster_salary.mean()) if n_with_salary else np.nan,
                "salary_coverage": float(n_with_salary / n_jobs) if n_jobs else 0.0,
                "top_skills": _top_skills_for_values(group[skill_column], n_skills=5) if skill_column else "",
                "centroid_drift_from_global": _cosine_distance(centroids[int(cluster_id)], temporal_centroid),
                "centroid_drift_from_previous_bin": drift_previous,
                "n_with_salary": n_with_salary,
                "n_with_skills": n_with_skills,
            }
        )
    return pd.DataFrame(rows, columns=METRIC_COLUMNS).sort_values(["time_bin", "cluster_id"], ignore_index=True)


def _currency_ok(df: pd.DataFrame) -> pd.Series:
    if "currency" not in df.columns:
        return pd.Series(True, index=df.index)
    currency = df["currency"].fillna("").astype(str).str.upper()
    return currency.eq("USD") | currency.eq("")


def compute_cluster_growth(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame(columns=["cluster_id", "cluster_label", "first_share", "last_share", "share_delta"])
    rows: list[dict[str, Any]] = []
    for cluster_id, group in metrics.sort_values("time_bin").groupby("cluster_id"):
        first = group.iloc[0]
        last = group.iloc[-1]
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_label": str(last["cluster_label"]),
                "first_share": float(first["share_jobs"]),
                "last_share": float(last["share_jobs"]),
                "share_delta": float(last["share_jobs"] - first["share_jobs"]),
            }
        )
    return pd.DataFrame(rows).sort_values("share_delta", ascending=False, ignore_index=True)


def compute_decay_inputs(
    df: pd.DataFrame,
    schema: dict[str, Any],
) -> tuple[pd.DataFrame, str | None]:
    """Return duration/event data when lifecycle columns support it."""
    posted = df["_posted_at"]
    if schema.get("closure_column"):
        closed = _parse_datetime(df[schema["closure_column"]])
        duration = (closed - posted).dt.total_seconds() / 86400.0
        event_observed = closed.notna()
        reason = None
    elif schema.get("last_seen_column"):
        last_seen = _parse_datetime(df[schema["last_seen_column"]])
        duration = (last_seen - posted).dt.total_seconds() / 86400.0
        event_observed = pd.Series(False, index=df.index)
        reason = None
    else:
        return pd.DataFrame(), "No closure, expiration, taken, or last_seen column was detected."

    out = pd.DataFrame(
        {
            "cluster_id": df["cluster_id"].astype(int),
            "duration_days": duration,
            "event_observed": event_observed.astype(bool),
        }
    )
    out = out[out["duration_days"].notna() & (out["duration_days"] >= 0)].copy()
    if out.empty:
        return out, "Lifecycle columns were detected, but no non-negative durations could be computed."
    return out, reason


def fit_exponential_decay_by_cluster(
    durations: pd.DataFrame,
    labels: pd.DataFrame,
    min_rows: int = 5,
    min_events: int = 1,
) -> pd.DataFrame:
    """Fit exponential MLE with right-censoring: lambda = events / total time at risk."""
    label_map = labels.set_index("cluster_id")["cluster_label"].to_dict()
    rows: list[dict[str, Any]] = []
    for cluster_id, group in durations.groupby("cluster_id", sort=True):
        n_rows = int(len(group))
        n_events = int(group["event_observed"].sum())
        n_censored = n_rows - n_events
        total_time = float(group["duration_days"].sum())
        warning = ""
        if n_rows < min_rows or n_events < min_events or total_time <= 0:
            lambda_hat = np.nan
            half_life = np.nan
            mean_lifetime = np.nan
            warning = f"insufficient data: n={n_rows}, events={n_events}"
        else:
            lambda_hat = float(n_events / total_time)
            half_life = float(math.log(2.0) / lambda_hat) if lambda_hat > 0 else np.nan
            mean_lifetime = float(1.0 / lambda_hat) if lambda_hat > 0 else np.nan
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_label": label_map.get(int(cluster_id), f"C{int(cluster_id):02d}"),
                "n_rows": n_rows,
                "n_events": n_events,
                "n_censored": n_censored,
                "total_time_at_risk_days": total_time,
                "lambda": lambda_hat,
                "half_life_days": half_life,
                "mean_lifetime_days": mean_lifetime,
                "warning": warning,
            }
        )
    return pd.DataFrame(rows).sort_values("cluster_id", ignore_index=True)


def _write_decay_not_available(path: Path, reason: str) -> None:
    path.write_text(
        "\n".join(
            [
                "# Decay Analysis Not Available",
                "",
                reason,
                "",
                "No survival or exponential decay plots were produced because the dataset does not support them.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_cluster_bubble_timeline(metrics: pd.DataFrame, labels: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    if metrics.empty:
        ax.text(0.5, 0.5, "No temporal cluster metrics available", ha="center", va="center")
        ax.axis("off")
    else:
        selected = _select_focus_clusters(labels, ["health", "technology", "sales", "construction"], max_clusters=4)
        data = metrics[metrics["cluster_id"].isin(selected["cluster_id"])].copy()
        data["_time"] = _time_bin_start(data["time_bin"])
        data = data[data["_time"].notna()]
        data = _complete_daily_focus_grid(data.sort_values(["cluster_id", "_time"]), selected)
        max_jobs = max(float(data["n_jobs"].max()), 1.0)
        sizes = 30 + 360 * np.sqrt(data["n_jobs"]) / math.sqrt(max_jobs)
        for _, cluster in selected.iterrows():
            subset = data[data["cluster_id"] == int(cluster["cluster_id"])].sort_values("_time")
            if subset.empty:
                continue
            ax.plot(subset["_time"], subset["share_jobs"], color=str(cluster["color"]), linewidth=1.2, alpha=0.45)
            ax.scatter(
                subset["_time"],
                subset["share_jobs"],
                s=sizes.loc[subset.index],
                color=str(cluster["color"]),
                alpha=0.78,
                edgecolors="#333333",
                linewidths=0.35,
                label=str(cluster["display_label"]),
            )
        ax.set_xlabel("Posting date")
        ax.set_ylabel("Posting share")
        ax.set_title("Daily Posting Share for Focus Clusters")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_cluster_share_timeseries(metrics: pd.DataFrame, path: Path, top_n: int = 8) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6))
    if metrics.empty:
        ax.text(0.5, 0.5, "No temporal cluster metrics available", ha="center", va="center")
        ax.axis("off")
    else:
        unique_bins = sorted(metrics["time_bin"].dropna().astype(str).unique())
        bin_to_idx = {time_bin: i for i, time_bin in enumerate(unique_bins)}
        top_clusters = (
            metrics.groupby(["cluster_id", "cluster_label"])["n_jobs"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )
        for _, row in top_clusters.iterrows():
            subset = metrics[metrics["cluster_id"] == row["cluster_id"]].sort_values("time_bin")
            x_values = subset["time_bin"].astype(str).map(bin_to_idx)
            ax.plot(x_values, subset["share_jobs"], marker="o", label=row["cluster_label"])
        ax.set_title(f"Participación de clusters en la ventana observada")
        ax.set_xlabel("Intervalo")
        ax.set_ylabel("Share de postings")
        if unique_bins:
            step = max(1, math.ceil(len(unique_bins) / 12))
            ticks = list(range(0, len(unique_bins), step))
            if ticks[-1] != len(unique_bins) - 1:
                ticks.append(len(unique_bins) - 1)
            ax.set_xticks(ticks)
            ax.set_xticklabels([unique_bins[i] for i in ticks])
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def prepare_cluster_trajectory_change_data(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    labels: pd.DataFrame,
    random_state: int,
    top_n: int = 5,
) -> pd.DataFrame:
    """Build top-N start/end projected centroid changes for plotting/tests."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "cluster_id",
                "cluster_label",
                "start_bin",
                "end_bin",
                "start_n",
                "end_n",
                "x_start",
                "y_start",
                "x_end",
                "y_end",
                "projected_distance",
            ]
        )
    projected, _ = _projection_2d(embeddings, random_state)
    label_map = labels.set_index("cluster_id")["cluster_label"].to_dict()
    top_clusters = df.groupby("cluster_id").size().sort_values(ascending=False).head(top_n).index.tolist()
    rows: list[dict[str, Any]] = []
    for cluster_id in top_clusters:
        cluster = df[df["cluster_id"] == cluster_id].sort_values("time_bin")
        bins = sorted(cluster["time_bin"].dropna().unique())
        if not bins:
            continue
        start_bin, end_bin = bins[0], bins[-1]
        start_group = cluster[cluster["time_bin"] == start_bin]
        end_group = cluster[cluster["time_bin"] == end_bin]
        start_point = projected[start_group["_embedding_index"].to_numpy(dtype=int)].mean(axis=0)
        end_point = projected[end_group["_embedding_index"].to_numpy(dtype=int)].mean(axis=0)
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_label": label_map.get(int(cluster_id), f"C{int(cluster_id):02d}"),
                "start_bin": str(start_bin),
                "end_bin": str(end_bin),
                "start_n": int(len(start_group)),
                "end_n": int(len(end_group)),
                "x_start": float(start_point[0]),
                "y_start": float(start_point[1]),
                "x_end": float(end_point[0]),
                "y_end": float(end_point[1]),
                "projected_distance": float(np.linalg.norm(end_point - start_point)),
            }
        )
    return pd.DataFrame(rows).sort_values("projected_distance", ascending=False, ignore_index=True)


def _projection_2d(embeddings: np.ndarray, random_state: int) -> tuple[np.ndarray, Any | None]:
    if embeddings.shape[1] >= 2:
        projector = TruncatedSVD(n_components=2, random_state=random_state)
        return projector.fit_transform(embeddings), projector
    x = embeddings[:, 0] if embeddings.shape[1] == 1 else np.zeros(len(embeddings))
    return np.column_stack([x, np.zeros(len(embeddings))]), None


def _write_cluster_semantic_trajectory(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    centroids: np.ndarray,
    labels: pd.DataFrame,
    path: Path,
    random_state: int,
    top_n: int = 5,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6.4))
    if df.empty:
        ax.text(0.5, 0.5, "No hay trayectoria temporal de clusters suficiente", ha="center", va="center")
        ax.axis("off")
    else:
        change = prepare_cluster_trajectory_change_data(df, embeddings, labels, random_state, top_n=top_n)
        colors = ["#2563eb", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]
        for idx, row in change.iterrows():
            color = colors[idx % len(colors)]
            ax.scatter(row["x_start"], row["y_start"], s=55 + 8 * math.sqrt(row["start_n"]), color=color, alpha=0.45)
            ax.scatter(row["x_end"], row["y_end"], s=75 + 8 * math.sqrt(row["end_n"]), color=color, edgecolor="white", linewidth=0.8)
            ax.annotate(
                "",
                xy=(row["x_end"], row["y_end"]),
                xytext=(row["x_start"], row["y_start"]),
                arrowprops={"arrowstyle": "->", "lw": 1.8, "color": color, "alpha": 0.9},
            )
            label = f"{row['cluster_label']} (Δ={row['projected_distance']:.2f}, n {row['start_n']}→{row['end_n']})"
            ax.plot([], [], color=color, marker="o", linestyle="-", label=label)
            ax.annotate(str(row["cluster_id"]), (row["x_end"], row["y_end"]), fontsize=8, color="#1e293b")
        if not change.empty:
            ax.text(
                0.02,
                0.98,
                f"Ventana: {change['start_bin'].min()} a {change['end_bin'].max()}",
                transform=ax.transAxes,
                va="top",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.9},
            )
        ax.set_title("Cambio semántico de clusters principales", fontsize=13, fontweight="bold", pad=18)
        ax.text(
            0.0,
            1.02,
            "Proyección 2D solo para visualización; el clustering se calcula en el espacio original.",
            transform=ax.transAxes,
            fontsize=9,
            color="#475569",
        )
        ax.set_xlabel("Proyección 1")
        ax.set_ylabel("Proyección 2")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _survival_curve_points(group: pd.DataFrame) -> pd.DataFrame:
    events = group[group["event_observed"]].sort_values("duration_days")
    if events.empty:
        return pd.DataFrame(columns=["duration_days", "survival"])
    n_at_risk = len(group)
    survival = 1.0
    rows = [{"duration_days": 0.0, "survival": 1.0}]
    for duration, event_group in events.groupby("duration_days"):
        censored_before = int(((group["duration_days"] < duration) & (~group["event_observed"])).sum())
        events_at_time = int(len(event_group))
        at_risk = max(n_at_risk - censored_before, 1)
        survival *= max(0.0, 1.0 - events_at_time / at_risk)
        rows.append({"duration_days": float(duration), "survival": float(survival)})
    return pd.DataFrame(rows)


def _duration_axis_scale(durations: pd.DataFrame) -> tuple[float, str]:
    max_duration = float(durations["duration_days"].max()) if not durations.empty else 0.0
    if max_duration <= 3.0:
        return 24.0, "Hours since posting"
    return 1.0, "Days since posting"


def _dense_survival_points(curve: pd.DataFrame, max_duration_days: float, n_points: int = 90) -> pd.DataFrame:
    if curve.empty:
        return curve
    event_x = curve["duration_days"].to_numpy(dtype=float)
    event_y = curve["survival"].to_numpy(dtype=float)
    grid_x = np.linspace(0.0, max(max_duration_days, float(event_x.max())), n_points)
    x_values = np.unique(np.concatenate([grid_x, event_x]))
    positions = np.searchsorted(event_x, x_values, side="right") - 1
    y_values = np.where(positions >= 0, event_y[np.clip(positions, 0, len(event_y) - 1)], 1.0)
    return pd.DataFrame({"duration_days": x_values, "survival": y_values})


def _write_survival_curves(durations: pd.DataFrame, labels: pd.DataFrame, path: Path, top_n: int = 5) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    selected = _select_focus_clusters(labels, ["health", "technology", "sales", "construction", "education"], max_clusters=top_n)
    selected_durations = durations[durations["cluster_id"].isin(selected["cluster_id"])]
    if selected_durations.empty:
        ax.text(0.5, 0.5, "No lifecycle durations for focus clusters", ha="center", va="center")
        ax.axis("off")
    else:
        scale, xlabel = _duration_axis_scale(selected_durations)
        max_duration = float(selected_durations["duration_days"].max())
        for _, cluster in selected.iterrows():
            cluster_id = int(cluster["cluster_id"])
            curve = _survival_curve_points(durations[durations["cluster_id"] == cluster_id])
            if curve.empty:
                continue
            dense = _dense_survival_points(curve, max_duration)
            ax.scatter(
                dense["duration_days"] * scale,
                dense["survival"],
                s=22,
                color=str(cluster["color"]),
                alpha=0.75,
                label=str(cluster["display_label"]),
            )
        ax.set_title("Observed Survival Points for Focus Clusters")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Estimated survival")
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_decay_fit(summary: pd.DataFrame, durations: pd.DataFrame, labels: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    selected = _select_focus_clusters(labels, ["health", "technology", "sales", "construction", "education"], max_clusters=5)
    selected_summary = selected.merge(summary, on="cluster_id", how="left", suffixes=("_selected", ""))
    selected_durations = durations[durations["cluster_id"].isin(selected["cluster_id"])]
    fitted = selected_summary.dropna(subset=["lambda"])
    if selected_durations.empty:
        ax.text(0.5, 0.5, "No lifecycle durations for focus clusters", ha="center", va="center")
        ax.axis("off")
    else:
        scale, xlabel = _duration_axis_scale(selected_durations)
        max_duration = max(float(selected_durations["duration_days"].max()), 1.0 / scale)
        x_days = np.linspace(0.0, max_duration, 220)
        for _, cluster in selected_summary.iterrows():
            cluster_id = int(cluster["cluster_id"])
            color = str(cluster["color"])
            label = str(cluster["display_label"])
            curve = _survival_curve_points(durations[durations["cluster_id"] == cluster_id])
            if not curve.empty:
                ax.scatter(curve["duration_days"] * scale, curve["survival"], s=18, color=color, alpha=0.35)
            lambda_hat = cluster.get("lambda")
            if pd.notna(lambda_hat):
                ax.plot(x_days * scale, np.exp(-float(lambda_hat) * x_days), color=color, linewidth=2.0, label=label)
            elif not curve.empty:
                ax.scatter([], [], color=color, label=f"{label} (no fit)")
        if fitted.empty:
            ax.text(0.5, 0.5, "No focus clusters had enough events for exponential fit", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Exponential Survival Fit for Focus Clusters")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Survival probability")
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _format_table(df: pd.DataFrame, columns: list[str], limit: int = 10) -> str:
    if df.empty:
        return "_No rows available._"
    table = df[columns].head(limit).copy()
    table = table.astype(str).apply(lambda column: column.str.replace("|", "\\|", regex=False))
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for package in ["numpy", "pandas", "sklearn", "matplotlib", "pyarrow"]:
        try:
            module = __import__(package)
            versions[package] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[package] = "not_importable"
    return versions


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return completed.stdout.strip()
    except Exception:
        return None


def _write_report(
    report_path: Path,
    manifest: dict[str, Any],
    labels: pd.DataFrame,
    metrics: pd.DataFrame,
    growth: pd.DataFrame,
    decay_summary: pd.DataFrame | None,
    plot_paths: dict[str, Path],
) -> None:
    top_clusters = labels.sort_values("n_jobs", ascending=False)
    rising = growth.sort_values("share_delta", ascending=False)
    declining = growth[growth["share_delta"] < 0].sort_values("share_delta", ascending=True)
    lines = [
        "# Temporal Cluster Analytics",
        "",
        "## Dataset",
        f"- Date range: {manifest['date_range']['min']} to {manifest['date_range']['max']}",
        f"- Rows input: {manifest['input_row_count']}",
        f"- Rows used: {manifest['selected_row_count']}",
        f"- Time bin: `{manifest['bin_size']}`",
        f"- Fixed clusters: {manifest['k_effective']}",
        "",
        "## Detected Schema",
        f"- Text columns: {', '.join(manifest['schema_mapping']['text_columns'])}",
        f"- Time column: `{manifest['schema_mapping']['time_column']}`",
        f"- Skills column: `{manifest['schema_mapping'].get('skill_column')}`",
        f"- Salary available: {manifest['salary_available']}",
        f"- Decay available: {manifest['decay_available']}",
        "",
        "## Top Clusters by Volume",
        _format_table(top_clusters, ["cluster_id", "cluster_label", "n_jobs"], 12),
        "",
        "## Top Growing Clusters",
        _format_table(rising, ["cluster_id", "cluster_label", "first_share", "last_share", "share_delta"], 8),
        "",
        "## Top Declining Clusters",
        _format_table(declining, ["cluster_id", "cluster_label", "first_share", "last_share", "share_delta"], 8),
        "",
        "## Salary Availability",
        f"- Rows with usable salary: {manifest['salary_summary']['n_with_salary']}",
        f"- Salary coverage: {manifest['salary_summary']['coverage']:.4f}",
        "",
        "## Decay Availability",
        f"- Available: {manifest['decay_available']}",
        f"- Reason/status: {manifest['decay_summary']}",
        "",
        "## Primary Plots",
        f"- Bubble timeline: `{plot_paths['bubble']}`",
        f"- Share timeseries: `{plot_paths['share']}`",
        f"- Semantic trajectory: `{plot_paths['trajectory']}`",
        "",
        "The bubble timeline focuses on four interpretable sectors selected from the fixed clusters: health, tech, sales, and construction. The x-axis is daily posting date, the y-axis is posting share, color identifies the sector, and bubble size is posting count.",
        "",
        "The share timeseries tracks the largest fixed clusters over time. Labels come from cluster terms and skills, not raw IDs.",
        "",
        "The survival plot uses marker-only survival estimates for the focus sectors. The exponential fit plot overlays observed survival points with fitted exponential survival curves for clusters with enough lifecycle events.",
        "",
        "The semantic trajectory plot is a 2D projection for interpretation only. Clustering is performed in the embedding space before projection.",
        "",
        "## Output Tables",
        f"- Metrics: `{manifest['output_paths']['cluster_time_metrics']}`",
        f"- Labels: `{manifest['output_paths']['cluster_labels_csv']}`",
        f"- Assignments: `{manifest['output_paths']['cluster_assignments']}`",
    ]
    if decay_summary is not None and not decay_summary.empty:
        lines.extend(["", "## Decay Summary", _format_table(decay_summary, ["cluster_id", "n_events", "n_censored", "lambda", "half_life_days"], 12)])
    lines.extend(
        [
            "",
            "## Limitations",
        ]
    )
    lines.extend([f"- {item}" for item in manifest["limitations"]])
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_temporal_clusters(
    input_path: Path,
    outdir: Path,
    bin_size: str = "D",
    k: int = 12,
    embedding: str = "tfidf_svd",
    max_rows: int | None = 100000,
    random_state: int = 42,
    command_used: str | None = None,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    embedding_batch_size: int = 8,
    time_column: str | None = None,
) -> TemporalClusterResult:
    """Run fixed temporal cluster analytics and write all report artifacts."""
    started_at = time.perf_counter()
    outdir.mkdir(parents=True, exist_ok=True)
    df_input = pd.read_parquet(input_path)
    schema = detect_temporal_cluster_schema(df_input, time_column=time_column)
    selected = _select_rows(df_input, max_rows=max_rows, random_state=random_state)
    selected["_embedding_index"] = np.arange(len(selected), dtype=int)
    selected["_analysis_text"] = _join_text_columns(selected, schema["text_columns"])
    selected["_posted_at"] = _parse_datetime(selected[schema["time_column"]])
    selected["time_bin"] = _time_bins(selected["_posted_at"], bin_size)

    embeddings, vectorizer, svd, tfidf, embedding_meta = build_temporal_cluster_embeddings(
        selected["_analysis_text"],
        embedding=embedding,
        random_state=random_state,
        embedding_model=embedding_model,
        embedding_batch_size=embedding_batch_size,
    )
    cluster_ids, centroids, _ = fit_fixed_clusters(embeddings, k=k, random_state=random_state)
    selected["cluster_id"] = cluster_ids
    labels = build_cluster_labels(
        selected,
        cluster_ids=cluster_ids,
        tfidf=tfidf,
        vectorizer=vectorizer,
        skill_column=schema["skill_column"],
    )
    metrics = compute_temporal_cluster_metrics(
        selected,
        embeddings=embeddings,
        centroids=centroids,
        labels=labels,
        skill_column=schema["skill_column"],
    )
    growth = compute_cluster_growth(metrics)

    labels_csv = outdir / "cluster_labels.csv"
    labels_json = outdir / "cluster_labels.json"
    labels_parquet = outdir / "cluster_labels.parquet"
    assignments_path = outdir / "cluster_assignments.parquet"
    metrics_path = outdir / "cluster_time_metrics.parquet"
    centroids_path = outdir / "global_cluster_centroids.npy"
    manifest_path = outdir / "manifest.json"
    report_path = outdir / "report.md"
    growth_path = outdir / "cluster_growth.parquet"

    labels.to_csv(labels_csv, index=False)
    labels.to_json(labels_json, orient="records", indent=2)
    labels.to_parquet(labels_parquet, index=False)
    assignment_columns = ["cluster_id", "time_bin", "_posted_at"]
    if "job_id" in selected.columns:
        assignment_columns.insert(0, "job_id")
    assignments = selected[assignment_columns].rename(columns={"_posted_at": "posted_at"})
    assignments.to_parquet(assignments_path, index=False)
    metrics.to_parquet(metrics_path, index=False)
    growth.to_parquet(growth_path, index=False)
    np.save(centroids_path, centroids)

    bubble_path = outdir / "cluster_bubble_timeline.png"
    share_path = outdir / "cluster_share_timeseries.png"
    trajectory_path = outdir / "cluster_semantic_trajectory.png"
    _write_cluster_bubble_timeline(metrics, labels, bubble_path)
    _write_cluster_share_timeseries(metrics, share_path)
    _write_cluster_semantic_trajectory(selected[selected["time_bin"].notna()], embeddings, centroids, labels, trajectory_path, random_state)

    generated_paths = [
        labels_csv,
        labels_json,
        labels_parquet,
        assignments_path,
        metrics_path,
        growth_path,
        centroids_path,
        bubble_path,
        share_path,
        trajectory_path,
    ]

    durations, decay_reason = compute_decay_inputs(selected, schema)
    decay_summary: pd.DataFrame | None = None
    if decay_reason is None and not durations.empty:
        decay_summary = fit_exponential_decay_by_cluster(durations, labels)
        decay_summary_path = outdir / "cluster_decay_summary.parquet"
        survival_path = outdir / "cluster_survival_curves.png"
        decay_fit_path = outdir / "cluster_decay_exponential_fit.png"
        decay_summary.to_parquet(decay_summary_path, index=False)
        _write_survival_curves(durations, labels, survival_path)
        _write_decay_fit(decay_summary, durations, labels, decay_fit_path)
        generated_paths.extend([decay_summary_path, survival_path, decay_fit_path])
        decay_available = True
        decay_status = "Lifecycle duration data detected; exponential right-censored fit was attempted by cluster."
    else:
        decay_not_available = outdir / "decay_not_available.md"
        _write_decay_not_available(decay_not_available, decay_reason or "No usable duration rows were available.")
        generated_paths.append(decay_not_available)
        decay_available = False
        decay_status = decay_reason or "No usable duration rows were available."

    annual_salary = compute_annual_salary(selected).where(_currency_ok(selected))
    n_with_salary = int(annual_salary.notna().sum())
    valid_dates = selected["_posted_at"].dropna()
    limitations = [
        "TF-IDF + SVD is a CPU-safe semantic baseline, not a deep language model.",
        "Clusters are fixed for the selected corpus; changing max rows or random seed can change the fitted global clusters.",
        "Salary metrics only use rows with usable annualized salary and USD/blank currency.",
        "Temporal bins with few postings can make cluster shares and drift noisy.",
    ]
    if embedding_meta.get("embedding_fallback_reason"):
        limitations.append("Sentence-transformer embeddings were requested but unavailable; the run fell back to TF-IDF + SVD.")
    manifest = {
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "input_path": str(input_path),
        "input_row_count": int(len(df_input)),
        "selected_row_count": int(len(selected)),
        "date_range": {
            "min": valid_dates.min().isoformat() if not valid_dates.empty else None,
            "max": valid_dates.max().isoformat() if not valid_dates.empty else None,
        },
        "bin_size": bin_size,
        "k_requested": int(k),
        "k_effective": int(len(labels)),
        "embedding_method": embedding_meta.get("embedding_method"),
        "requested_embedding": embedding,
        "random_seed": int(random_state),
        "command_used": command_used or "python -m jobsrec temporal-clusters",
        "git_commit": _git_commit(),
        "package_versions": _package_versions(),
        "schema_mapping": schema,
        "limitations": limitations,
        "salary_available": bool(n_with_salary > 0),
        "salary_summary": {
            "n_with_salary": n_with_salary,
            "coverage": float(n_with_salary / len(selected)) if len(selected) else 0.0,
        },
        "decay_available": bool(decay_available),
        "decay_summary": decay_status,
        "runtime_seconds": float(time.perf_counter() - started_at),
        "embedding": embedding_meta,
        "output_paths": {
            "cluster_time_metrics": str(metrics_path),
            "cluster_labels_csv": str(labels_csv),
            "cluster_labels_json": str(labels_json),
            "cluster_labels_parquet": str(labels_parquet),
            "cluster_assignments": str(assignments_path),
            "global_cluster_centroids": str(centroids_path),
            "report": str(report_path),
            "manifest": str(manifest_path),
        },
        "generated_files": [str(path) for path in generated_paths],
    }
    _write_report(
        report_path,
        manifest=manifest,
        labels=labels,
        metrics=metrics,
        growth=growth,
        decay_summary=decay_summary,
        plot_paths={"bubble": bubble_path, "share": share_path, "trajectory": trajectory_path},
    )
    generated_paths.extend([report_path, manifest_path])
    manifest["generated_files"] = [str(path) for path in generated_paths]
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    return TemporalClusterResult(
        output_dir=outdir,
        manifest_path=manifest_path,
        report_path=report_path,
        metrics_path=metrics_path,
        generated_files=[str(path) for path in generated_paths],
        manifest=manifest,
    )
