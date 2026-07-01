"""Fast temporal trend prototype for silver job postings."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


TEXT_COLUMN = "job_card_text"
SKILL_COLUMN = "skills_text"
TEMPORAL_COLUMNS = [
    "first_seen_at",
    "last_seen_at",
    "listed_time",
    "original_listed_time",
    "expiry",
    "closed_time",
]
TIME_COLUMN_INTERPRETATIONS = {
    "first_seen_at": "first scraper observation time",
    "last_seen_at": "last scraper observation time",
    "listed_time": "canonical posting-listing time",
    "original_listed_time": "wider but sparse original listing time",
    "expiry": "expiration lifecycle time, not posting-demand coverage",
    "closed_time": "posting close lifecycle time, sparse when present",
}
SALARY_WEIGHT_CLIP_MIN = 0.25
SALARY_WEIGHT_CLIP_MAX = 4.0
REQUIRED_DRIFT_COLUMNS = [
    "month",
    "previous_month",
    "cosine_similarity",
    "centroid_drift",
    "jobs_in_month",
]
SEMANTIC_DRIFT_COLUMNS = [
    "month_from",
    "month_to",
    "n_from",
    "n_to",
    "representation",
    "embedding_backend",
    "embedding_model",
    "cosine_similarity",
    "cosine_distance",
]


@dataclass(frozen=True)
class TemporalDemoResult:
    """Paths and manifest for a completed temporal demo run."""

    weekly_drift_path: Path
    skill_growth_path: Path
    manifest_path: Path
    report_path: Path
    generated_files: list[str]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class TemporalAuditResult:
    """Paths and summary for a completed temporal audit."""

    report_path: Path
    summary_path: Path
    monthly_counts_path: Path
    weekly_counts_path: Path
    daily_counts_path: Path
    temporal_coverage_path: Path
    summary: dict[str, Any]


def parse_listed_time(df: pd.DataFrame) -> pd.Series:
    """Parse listed_time with pandas coercion semantics."""
    return parse_time_column(df, "listed_time")


def parse_time_column(df: pd.DataFrame, time_column: str) -> pd.Series:
    """Parse a temporal column with pandas coercion and numeric epoch fallback."""
    if time_column not in df.columns:
        raise ValueError(f"Input data must contain a '{time_column}' column.")
    parsed = pd.to_datetime(df[time_column], errors="coerce")

    numeric = pd.to_numeric(df[time_column], errors="coerce")
    if numeric.notna().any() and parsed.notna().any():
        parsed_years = parsed.dropna().dt.year
        if not parsed_years.empty and parsed_years.median() < 1990:
            for unit in ("ms", "s"):
                reparsed = pd.to_datetime(numeric, errors="coerce", unit=unit)
                years = reparsed.dropna().dt.year
                if not years.empty and 1990 <= years.median() <= 2100:
                    return reparsed
    return parsed


def _period_frequency(time_bin: str) -> str:
    if time_bin.upper() == "H":
        return "h"
    return time_bin.upper()


def add_month_bucket(
    df: pd.DataFrame,
    time_column: str = "listed_time",
    time_bin: str = "M",
) -> pd.DataFrame:
    """Return a copy with parsed listed time and a period bucket.

    The output column is still named ``month`` for compatibility with the
    existing analytics code, but it may contain daily or hourly period labels.
    """
    out = df.copy()
    out["_listed_time_parsed"] = parse_time_column(out, time_column)
    out["month"] = out["_listed_time_parsed"].dt.to_period(_period_frequency(time_bin)).astype("string")
    out["_time_column"] = time_column
    out["_time_bin"] = time_bin.upper()
    return out


def _non_empty_mask(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().ne("")


def build_reliability_assessment(
    monthly_counts: dict[str, int],
    min_months: int = 6,
    min_rows_per_month: int = 1000,
) -> dict[str, Any]:
    """Classify temporal support and produce reportable warnings."""
    months = sorted(monthly_counts)
    n_months = len(months)
    low_support_months = {
        month: int(count)
        for month, count in monthly_counts.items()
        if int(count) < min_rows_per_month
    }
    warnings: list[str] = []
    known_limitations: list[str] = []

    if n_months < 2:
        label = "demo_only"
        warnings.append("Fewer than two valid months are available; drift comparison is not supported.")
    elif n_months < min_months:
        label = "limited_temporal_coverage"
        warnings.append(
            f"Only {n_months} months are available; this is a temporal comparison, not a stable trend."
        )
    else:
        label = "sufficient_temporal_coverage"

    if n_months == 2:
        warnings.append("Only 2 months are available; this is a two-bucket comparison, not a trend.")
        known_limitations.append("Do not describe two-month output as market evolution.")
    if low_support_months:
        warnings.append(
            f"Months below {min_rows_per_month} rows may make first/last or consecutive comparisons noisy: "
            + ", ".join(f"{month}={count}" for month, count in sorted(low_support_months.items()))
        )
    if months and (months[0] in low_support_months or months[-1] in low_support_months):
        warnings.append("First/last comparison is likely noisy because an endpoint month has low support.")

    return {
        "label": label,
        "n_months": n_months,
        "min_months_required": int(min_months),
        "min_rows_per_month": int(min_rows_per_month),
        "low_support_months": low_support_months,
        "warnings": warnings,
        "known_limitations": known_limitations,
    }


def compute_temporal_column_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize parse coverage and month support for all temporal columns."""
    rows: list[dict[str, Any]] = []
    for column in TEMPORAL_COLUMNS:
        if column not in df.columns:
            continue
        parsed = parse_time_column(df, column)
        valid = parsed.notna()
        months = parsed[valid].dt.to_period("M").astype(str).value_counts().sort_index()
        reliability = build_reliability_assessment({str(k): int(v) for k, v in months.items()})
        rows.append(
            {
                "time_column": column,
                "interpretation": TIME_COLUMN_INTERPRETATIONS.get(column, "temporal column"),
                "total_rows": int(len(df)),
                "valid_rows": int(valid.sum()),
                "invalid_or_missing_rows": int((~valid).sum()),
                "parse_success_rate": float(valid.mean()) if len(df) else 0.0,
                "min_date": parsed[valid].min().isoformat() if valid.any() else None,
                "max_date": parsed[valid].max().isoformat() if valid.any() else None,
                "number_of_months": int(len(months)),
                "rows_per_month": json.dumps({str(k): int(v) for k, v in months.items()}),
                "reliability_label": reliability["label"],
                "warnings": json.dumps(reliability["warnings"]),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "time_column",
            "interpretation",
            "total_rows",
            "valid_rows",
            "invalid_or_missing_rows",
            "parse_success_rate",
            "min_date",
            "max_date",
            "number_of_months",
            "rows_per_month",
            "reliability_label",
            "warnings",
        ],
    )


def compute_temporal_audit(
    df: pd.DataFrame,
    input_path: Path | str = "",
    time_column: str = "listed_time",
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute full-dataset temporal audit tables and summary metadata."""
    audited = add_month_bucket(df, time_column=time_column)
    total_rows = int(len(audited))
    valid = audited.dropna(subset=["_listed_time_parsed"]).copy()
    n_valid = int(len(valid))
    n_invalid = total_rows - n_valid
    parse_rate = float(n_valid / total_rows) if total_rows else 0.0

    if TEXT_COLUMN not in audited.columns:
        audited[TEXT_COLUMN] = ""
    if SKILL_COLUMN not in audited.columns:
        audited[SKILL_COLUMN] = ""

    monthly = (
        valid.groupby("month", dropna=True)
        .agg(
            rows=("month", "size"),
            job_card_text_non_empty=(TEXT_COLUMN, lambda s: int(_non_empty_mask(s).sum())),
            skills_text_non_empty=(SKILL_COLUMN, lambda s: int(_non_empty_mask(s).sum())),
        )
        .reset_index()
        .sort_values("month", ignore_index=True)
    )
    if monthly.empty:
        monthly = pd.DataFrame(
            columns=["month", "rows", "job_card_text_non_empty", "skills_text_non_empty"]
        )
    monthly["job_card_text_coverage"] = np.where(
        monthly["rows"].astype(int) > 0,
        monthly["job_card_text_non_empty"] / monthly["rows"],
        0.0,
    )
    monthly["skills_text_coverage"] = np.where(
        monthly["rows"].astype(int) > 0,
        monthly["skills_text_non_empty"] / monthly["rows"],
        0.0,
    )

    weekly = valid.assign(week=valid["_listed_time_parsed"].dt.to_period("W").astype("string"))
    weekly_counts = weekly.groupby("week").size().rename("rows").reset_index()
    daily = valid.assign(day=valid["_listed_time_parsed"].dt.date.astype(str))
    daily_counts = daily.groupby("day").size().rename("rows").reset_index()

    monthly_counts = {str(row["month"]): int(row["rows"]) for _, row in monthly.iterrows()}
    reliability = build_reliability_assessment(monthly_counts)
    warnings = list(reliability["warnings"])
    if reliability["n_months"] < 6:
        warnings.append("Available months < 6; temporal outputs should be labeled limited.")

    temporal_column_coverage = compute_temporal_column_coverage(df)
    coverage_records = temporal_column_coverage.to_dict(orient="records")
    summary: dict[str, Any] = {
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "input_path": str(input_path),
        "time_column": time_column,
        "time_column_interpretation": TIME_COLUMN_INTERPRETATIONS.get(time_column, "temporal column"),
        "total_rows": total_rows,
        "valid_time_rows": n_valid,
        "invalid_or_missing_time_rows": n_invalid,
        "valid_listed_time_rows": n_valid,
        "invalid_or_missing_listed_time_rows": n_invalid,
        "parse_success_rate": parse_rate,
        "min_date": valid["_listed_time_parsed"].min().isoformat() if n_valid else None,
        "max_date": valid["_listed_time_parsed"].max().isoformat() if n_valid else None,
        "number_of_months": int(len(monthly_counts)),
        "rows_per_month": monthly_counts,
        "warnings": warnings,
        "reliability_label": reliability["label"],
        "reliability": reliability,
        "temporal_column_coverage": coverage_records,
        "deferred_tasks": [
            "Bootstrap stability for drift and skill growth is deferred; add a small explicit seed list before using confidence intervals."
        ],
    }
    return summary, monthly, weekly_counts, daily_counts, temporal_column_coverage


def _write_temporal_audit_report(
    report_path: Path,
    summary: dict[str, Any],
    monthly: pd.DataFrame,
    temporal_column_coverage: pd.DataFrame,
    command_used: str,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Temporal Audit",
        "",
        "## Command",
        f"`{command_used}`",
        "",
        "## Summary",
        f"- Input path: `{summary['input_path']}`",
        f"- Primary time column: `{summary['time_column']}` ({summary['time_column_interpretation']})",
        f"- Total rows: {summary['total_rows']}",
        f"- Valid selected-time rows: {summary['valid_time_rows']}",
        f"- Invalid/missing selected-time rows: {summary['invalid_or_missing_time_rows']}",
        f"- Parse success rate: {summary['parse_success_rate']:.4f}",
        f"- Min date: {summary['min_date']}",
        f"- Max date: {summary['max_date']}",
        f"- Number of months: {summary['number_of_months']}",
        f"- Reliability label: `{summary['reliability_label']}`",
        "",
        "## Warnings",
    ]
    warnings = summary.get("warnings", [])
    lines.extend([f"- {warning}" for warning in warnings] if warnings else ["- None"])
    lines.extend(
        [
            "",
            "## Monthly Counts",
            _format_table(
                monthly,
                ["month", "rows", "job_card_text_coverage", "skills_text_coverage"],
                100,
            ),
            "",
            "## Temporal Column Coverage",
            _format_table(
                temporal_column_coverage,
                [
                    "time_column",
                    "interpretation",
                    "valid_rows",
                    "number_of_months",
                    "reliability_label",
                ],
                20,
            ),
            "",
            "## Interpretation",
            "This audit describes temporal coverage only. It does not establish market evolution.",
            "",
            "## Deferred Tasks",
            "- Bootstrap stability for drift and skill growth is deferred; add a small explicit seed list before using confidence intervals.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_temporal_audit(
    silver_path: Path,
    output_dir: Path,
    command_used: str | None = None,
    time_column: str = "listed_time",
) -> TemporalAuditResult:
    """Run a full-dataset temporal audit and write report, JSON, and Parquet outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(silver_path)
    summary, monthly, weekly, daily, temporal_column_coverage = compute_temporal_audit(
        df,
        input_path=silver_path,
        time_column=time_column,
    )

    report_path = output_dir / "report.md"
    summary_path = output_dir / "summary.json"
    monthly_counts_path = output_dir / "monthly_counts.parquet"
    weekly_counts_path = output_dir / "weekly_counts.parquet"
    daily_counts_path = output_dir / "daily_counts.parquet"
    temporal_coverage_path = output_dir / "temporal_column_coverage.parquet"

    summary["output_paths"] = {
        "report": str(report_path),
        "summary": str(summary_path),
        "monthly_counts": str(monthly_counts_path),
        "weekly_counts": str(weekly_counts_path),
        "daily_counts": str(daily_counts_path),
        "temporal_column_coverage": str(temporal_coverage_path),
    }
    summary["command"] = command_used or "python -m jobsrec.cli temporal-audit"

    monthly.to_parquet(monthly_counts_path, index=False)
    weekly.to_parquet(weekly_counts_path, index=False)
    daily.to_parquet(daily_counts_path, index=False)
    temporal_column_coverage.to_parquet(temporal_coverage_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_temporal_audit_report(report_path, summary, monthly, temporal_column_coverage, summary["command"])

    return TemporalAuditResult(
        report_path=report_path,
        summary_path=summary_path,
        monthly_counts_path=monthly_counts_path,
        weekly_counts_path=weekly_counts_path,
        daily_counts_path=daily_counts_path,
        temporal_coverage_path=temporal_coverage_path,
        summary=summary,
    )


def sample_jobs(
    df: pd.DataFrame,
    sample_size: int,
    sampling_mode: str = "temporal-stride",
    time_column: str = "listed_time",
    time_bin: str = "M",
) -> pd.DataFrame:
    """Sample rows for fast temporal analysis."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    if sampling_mode not in {"temporal-stride", "random", "head"}:
        raise ValueError(f"Unsupported sampling_mode: {sampling_mode}")

    if sampling_mode == "head":
        return add_month_bucket(df.head(sample_size), time_column=time_column).reset_index(drop=True)

    if sampling_mode == "random":
        n = min(sample_size, len(df))
        return add_month_bucket(df.sample(n=n, random_state=42), time_column=time_column).reset_index(drop=True)

    temporal = add_month_bucket(df, time_column=time_column, time_bin=time_bin)
    valid = temporal.dropna(subset=["_listed_time_parsed"]).sort_values("_listed_time_parsed")
    if valid.empty:
        return valid.reset_index(drop=True)

    months = sorted(valid["month"].dropna().unique().tolist())
    if sample_size >= len(months):
        mandatory = valid.groupby("month", sort=True, group_keys=False).head(1)
        remaining = valid.drop(index=mandatory.index)
        remaining_slots = sample_size - len(mandatory)
        if remaining_slots > 0 and not remaining.empty:
            stride = max(1, math.floor(len(remaining) / remaining_slots))
            selected = pd.concat([mandatory, remaining.iloc[::stride].head(remaining_slots)])
        else:
            selected = mandatory
        selected = selected.sort_values("_listed_time_parsed").head(sample_size).copy()
    else:
        stride = max(1, math.floor(len(valid) / sample_size))
        selected = valid.iloc[::stride].head(sample_size).copy()
        last_row = valid.tail(1)
        if not selected.empty and selected["month"].iloc[-1] != last_row["month"].iloc[0]:
            selected = pd.concat([selected.iloc[:-1], last_row], ignore_index=False)
    return selected.reset_index(drop=True)


def _split_skills(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    delimiter = ";" if ";" in text else ","
    return [" ".join(part.lower().split()) for part in text.split(delimiter) if part.strip()]


def _safe_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str)


def _vectorize_texts(texts: pd.Series) -> tuple[Any, str, bool]:
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=1)
    tfidf = vectorizer.fit_transform(_safe_text(texts))
    n_components = min(50, max(0, min(tfidf.shape) - 1))
    if n_components >= 2:
        try:
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            reduced = svd.fit_transform(tfidf)
            return reduced, "tfidf+svd", True
        except Exception:
            return tfidf, "tfidf", False
    return tfidf, "tfidf", False


def _build_embedding_backend(backend_name: str, model_name: str, device: str) -> Any:
    if backend_name == "mock":
        from jobsrec.embeddings.mock import MockEmbeddingBackend

        return MockEmbeddingBackend(model_name=model_name)
    if backend_name in {"existing_qwen3", "qwen3"}:
        try:
            from jobsrec.embeddings.qwen3 import Qwen3EmbeddingBackend

            return Qwen3EmbeddingBackend(model_name=model_name, device=device)
        except Exception as exc:
            raise RuntimeError(
                "Unable to initialize the existing Qwen3 backend. Use "
                "`--embedding-backend mock` for CPU-safe smoke tests, or install/configure "
                "sentence-transformers and run a small explicit Qwen3 smoke command."
            ) from exc
    raise ValueError(f"Unsupported embedding backend: {backend_name}")


def _embedding_cache_path(
    cache_dir: Path,
    texts: list[str],
    backend_name: str,
    model_name: str,
    batch_size: int,
    device: str,
) -> Path:
    hasher = hashlib.sha256()
    hasher.update(backend_name.encode("utf-8"))
    hasher.update(model_name.encode("utf-8"))
    hasher.update(str(batch_size).encode("utf-8"))
    hasher.update(device.encode("utf-8"))
    for text in texts:
        hasher.update(b"\0")
        hasher.update(text.encode("utf-8", errors="replace"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{hasher.hexdigest()[:24]}.npy"


def _load_or_compute_embeddings(
    texts: pd.Series,
    backend_name: str,
    model_name: str,
    batch_size: int,
    device: str,
    cache_dir: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    safe_texts = _safe_text(texts).tolist()
    cache_path = _embedding_cache_path(cache_dir, safe_texts, backend_name, model_name, batch_size, device)
    cache_hit = cache_path.exists()
    if cache_hit:
        embeddings = np.load(cache_path)
    else:
        backend = _build_embedding_backend(backend_name, model_name, device)
        embeddings = backend.encode_texts(safe_texts, batch_size=batch_size)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        np.save(cache_path, embeddings)

    if embeddings.ndim != 2 or embeddings.shape[0] != len(safe_texts):
        raise RuntimeError("Embedding backend returned an invalid matrix shape.")
    return embeddings, {
        "embedding_cache_path": str(cache_path),
        "embedding_cache_hit": bool(cache_hit),
        "embedding_rows": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
    }


def compute_semantic_centroid_drift(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    embedding_backend: str,
    embedding_model: str,
    centroids_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute normalized monthly embedding centroids and consecutive drift."""
    if df.empty:
        empty_drift = pd.DataFrame(
            columns=SEMANTIC_DRIFT_COLUMNS
            + ["month", "previous_month", "centroid_drift", "jobs_in_month"]
        )
        empty_meta = pd.DataFrame(
            columns=[
                "month",
                "n_jobs",
                "representation",
                "embedding_backend",
                "embedding_model",
                "vector_dim",
                "centroid_storage_path",
            ]
        )
        np.save(centroids_path, np.empty((0, 0), dtype=np.float32))
        return empty_drift, empty_meta

    if embeddings.shape[0] != len(df):
        raise ValueError("Embedding row count must match temporal rows.")

    months = sorted(df["month"].dropna().unique().tolist())
    centroid_rows: list[np.ndarray] = []
    metadata_rows: list[dict[str, Any]] = []
    volumes = df.groupby("month").size().to_dict()
    for month in months:
        mask = df["month"].to_numpy() == month
        centroid = embeddings[mask].mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        centroid_rows.append(centroid.astype(np.float32))
        metadata_rows.append(
            {
                "month": month,
                "n_jobs": int(volumes.get(month, 0)),
                "representation": "semantic_embeddings",
                "embedding_backend": embedding_backend,
                "embedding_model": embedding_model,
                "vector_dim": int(embeddings.shape[1]),
                "centroid_storage_path": str(centroids_path),
            }
        )

    centroid_matrix = np.vstack(centroid_rows) if centroid_rows else np.empty((0, embeddings.shape[1]))
    centroids_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(centroids_path, centroid_matrix)

    drift_rows: list[dict[str, Any]] = []
    for i in range(1, len(months)):
        month_from = months[i - 1]
        month_to = months[i]
        similarity = float(np.dot(centroid_matrix[i - 1], centroid_matrix[i]))
        distance = 1.0 - similarity
        drift_rows.append(
            {
                "month_from": month_from,
                "month_to": month_to,
                "n_from": int(volumes.get(month_from, 0)),
                "n_to": int(volumes.get(month_to, 0)),
                "representation": "semantic_embeddings",
                "embedding_backend": embedding_backend,
                "embedding_model": embedding_model,
                "cosine_similarity": similarity,
                "cosine_distance": distance,
                "month": month_to,
                "previous_month": month_from,
                "centroid_drift": distance,
                "jobs_in_month": int(volumes.get(month_to, 0)),
            }
        )

    drift = pd.DataFrame(
        drift_rows,
        columns=SEMANTIC_DRIFT_COLUMNS
        + ["month", "previous_month", "centroid_drift", "jobs_in_month"],
    )
    return drift, pd.DataFrame(metadata_rows)


def compute_annual_salary(df: pd.DataFrame) -> pd.Series:
    """Return annualized salary values using normalized salary, median, then range midpoint."""
    salary = pd.Series(np.nan, index=df.index, dtype="float64")
    if "normalized_salary" in df.columns:
        normalized = pd.to_numeric(df["normalized_salary"], errors="coerce")
        salary = salary.fillna(normalized)

    pay_period = df.get("pay_period", pd.Series("", index=df.index)).fillna("").astype(str).str.upper()
    multipliers = pay_period.map(
        {
            "YEARLY": 1.0,
            "MONTHLY": 12.0,
            "WEEKLY": 52.0,
            "BIWEEKLY": 26.0,
            "HOURLY": 2080.0,
        }
    )
    if "med_salary" in df.columns:
        med = pd.to_numeric(df["med_salary"], errors="coerce") * multipliers
        salary = salary.fillna(med)
    if "min_salary" in df.columns and "max_salary" in df.columns:
        min_salary = pd.to_numeric(df["min_salary"], errors="coerce")
        max_salary = pd.to_numeric(df["max_salary"], errors="coerce")
        midpoint = ((min_salary + max_salary) / 2.0) * multipliers
        salary = salary.fillna(midpoint)
    salary = salary.where(salary > 0)
    return salary


def prepare_salary_weights(
    df: pd.DataFrame,
    currency_filter: str = "USD",
    clip_min: float = SALARY_WEIGHT_CLIP_MIN,
    clip_max: float = SALARY_WEIGHT_CLIP_MAX,
) -> tuple[pd.Series, pd.DataFrame, dict[str, Any]]:
    """Create robust salary weights normalized within month."""
    annual_salary = compute_annual_salary(df)
    currency = df.get("currency", pd.Series("", index=df.index)).fillna("").astype(str).str.upper()
    if currency_filter:
        currency_ok = currency.eq(currency_filter.upper())
    else:
        currency_ok = pd.Series(True, index=df.index)
    usable = annual_salary.notna() & currency_ok & df["month"].notna()
    diagnostics = pd.DataFrame(
        {
            "job_id": df.get("job_id", pd.Series(np.nan, index=df.index)),
            "month": df["month"],
            "annual_salary": annual_salary,
            "currency": currency,
            "salary_usable": usable,
        }
    )
    weights = pd.Series(np.nan, index=df.index, dtype="float64")
    raw = np.log1p(annual_salary[usable])
    for month, values in raw.groupby(df.loc[usable, "month"]):
        mean = values.mean()
        if pd.isna(mean) or mean <= 0:
            continue
        weights.loc[values.index] = (values / mean).clip(lower=clip_min, upper=clip_max)
    diagnostics["salary_weight"] = weights

    summary = {
        "salary_weight_strategy": "normalized_salary_else_annualized_median_else_annualized_midpoint_log1p_month_mean_normalized_clipped",
        "salary_currency_filter": currency_filter,
        "salary_weight_clip_min": float(clip_min),
        "salary_weight_clip_max": float(clip_max),
        "salary_rows_used": int(weights.notna().sum()),
        "salary_coverage": float(weights.notna().mean()) if len(weights) else 0.0,
        "salary_non_usd_excluded": int((annual_salary.notna() & ~currency_ok).sum()),
        "salary_missing_or_unusable": int((~usable).sum()),
    }
    return weights, diagnostics, summary


def compute_salary_weighted_centroid_drift(
    df: pd.DataFrame,
    vectors: Any,
    representation: str,
    time_column: str,
    output_dir: Path,
    embedding_backend: str | None = None,
    embedding_model: str | None = None,
    currency_filter: str = "USD",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], Path]:
    """Compute normalized monthly centroids weighted by robust salary weights."""
    weights, diagnostics, salary_summary = prepare_salary_weights(df, currency_filter=currency_filter)
    usable = weights.notna()
    months = sorted(df.loc[usable, "month"].dropna().unique().tolist())
    vector_dim = int(vectors.shape[1]) if len(df) else 0
    centroids_path = output_dir / "weekly_centroids_salary_weighted.npy"

    centroid_rows: list[np.ndarray] = []
    metadata_rows: list[dict[str, Any]] = []
    total_volumes = df.groupby("month").size().to_dict()
    salary_volumes = df.loc[usable].groupby("month").size().to_dict()
    for month in months:
        mask = (df["month"].to_numpy() == month) & usable.to_numpy()
        month_weights = weights.loc[mask].to_numpy(dtype="float64")
        if len(month_weights) == 0 or month_weights.sum() <= 0:
            continue
        if sparse.issparse(vectors):
            centroid = np.asarray(vectors[mask].multiply(month_weights[:, None]).sum(axis=0)).ravel()
            centroid = centroid / month_weights.sum()
        else:
            dense = np.asarray(vectors)
            centroid = np.average(dense[mask], axis=0, weights=month_weights)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        centroid_rows.append(centroid.astype(np.float32))
        n_total = int(total_volumes.get(month, 0))
        n_salary = int(salary_volumes.get(month, 0))
        metadata_rows.append(
            {
                "month": month,
                "n_jobs": n_total,
                "n_salary_jobs": n_salary,
                "salary_coverage": float(n_salary / n_total) if n_total else 0.0,
                "representation": representation,
                "time_column": time_column,
                "centroid_weighting": "salary",
                "embedding_backend": embedding_backend,
                "embedding_model": embedding_model,
                "vector_dim": vector_dim,
                "centroid_storage_path": str(centroids_path),
            }
        )

    centroid_matrix = np.vstack(centroid_rows) if centroid_rows else np.empty((0, vector_dim), dtype=np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(centroids_path, centroid_matrix)

    rows: list[dict[str, Any]] = []
    for i in range(1, len(months)):
        month_from = months[i - 1]
        month_to = months[i]
        similarity = float(np.dot(centroid_matrix[i - 1], centroid_matrix[i]))
        n_from = int(total_volumes.get(month_from, 0))
        n_to = int(total_volumes.get(month_to, 0))
        n_salary_from = int(salary_volumes.get(month_from, 0))
        n_salary_to = int(salary_volumes.get(month_to, 0))
        rows.append(
            {
                "month_from": month_from,
                "month_to": month_to,
                "n_from": n_from,
                "n_to": n_to,
                "n_salary_from": n_salary_from,
                "n_salary_to": n_salary_to,
                "salary_coverage_from": float(n_salary_from / n_from) if n_from else 0.0,
                "salary_coverage_to": float(n_salary_to / n_to) if n_to else 0.0,
                "representation": representation,
                "time_column": time_column,
                "centroid_weighting": "salary",
                "cosine_similarity": similarity,
                "cosine_distance": 1.0 - similarity,
                "month": month_to,
                "previous_month": month_from,
                "centroid_drift": 1.0 - similarity,
                "jobs_in_month": n_to,
            }
        )
    drift = pd.DataFrame(
        rows,
        columns=[
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
            "month",
            "previous_month",
            "centroid_drift",
            "jobs_in_month",
        ],
    )
    return drift, pd.DataFrame(metadata_rows), diagnostics, salary_summary, centroids_path


def compute_centroid_drift(df: pd.DataFrame, vectors: Any) -> pd.DataFrame:
    """Compute normalized monthly centroids and consecutive-month drift."""
    if df.empty:
        return pd.DataFrame(columns=REQUIRED_DRIFT_COLUMNS)

    months = sorted(df["month"].dropna().unique().tolist())
    centroids: list[Any] = []
    volumes = df.groupby("month").size().to_dict()
    for month in months:
        mask = df["month"].to_numpy() == month
        if sparse.issparse(vectors):
            centroid = vectors[mask].mean(axis=0)
            centroid = np.asarray(centroid)
        else:
            centroid = np.asarray(vectors[mask]).mean(axis=0, keepdims=True)
        centroids.append(centroid)

    centroid_matrix = np.vstack(centroids)
    centroid_matrix = normalize(centroid_matrix)
    rows: list[dict[str, Any]] = []
    for i, month in enumerate(months):
        if i == 0:
            similarity = np.nan
            drift = np.nan
            previous_month = None
        else:
            similarity = float(np.dot(centroid_matrix[i - 1], centroid_matrix[i]))
            drift = 1.0 - similarity
            previous_month = months[i - 1]
        rows.append(
            {
                "month": month,
                "previous_month": previous_month,
                "cosine_similarity": similarity,
                "centroid_drift": drift,
                "jobs_in_month": int(volumes.get(month, 0)),
            }
        )
    return pd.DataFrame(rows, columns=REQUIRED_DRIFT_COLUMNS)


def compute_skill_growth(df: pd.DataFrame) -> pd.DataFrame:
    """Compare first and last selected months by monthly skill share."""
    valid = df.dropna(subset=["month"]).copy()
    if valid.empty:
        return pd.DataFrame(
            columns=[
                "skill",
                "first_month",
                "last_month",
                "first_count",
                "last_count",
                "first_share",
                "last_share",
                "share_delta",
                "trend",
            ]
        )

    months = sorted(valid["month"].unique().tolist())
    first_month, last_month = months[0], months[-1]
    rows: list[dict[str, Any]] = []
    for _, row in valid.iterrows():
        for skill in _split_skills(row.get(SKILL_COLUMN)):
            rows.append({"month": row["month"], "skill": skill})
    counts = pd.DataFrame(rows)
    if counts.empty:
        return pd.DataFrame(columns=["skill", "first_month", "last_month", "share_delta", "trend"])

    monthly = counts.groupby(["month", "skill"]).size().rename("count").reset_index()
    totals = monthly.groupby("month")["count"].sum().rename("total")
    monthly = monthly.merge(totals, on="month")
    monthly["share"] = monthly["count"] / monthly["total"]

    first = monthly[monthly["month"] == first_month][["skill", "count", "share"]].rename(
        columns={"count": "first_count", "share": "first_share"}
    )
    last = monthly[monthly["month"] == last_month][["skill", "count", "share"]].rename(
        columns={"count": "last_count", "share": "last_share"}
    )
    growth = first.merge(last, on="skill", how="outer").fillna(0)
    growth["first_month"] = first_month
    growth["last_month"] = last_month
    growth["share_delta"] = growth["last_share"] - growth["first_share"]
    growth["trend"] = np.where(growth["share_delta"] >= 0, "rising", "declining")
    return growth[
        [
            "skill",
            "first_month",
            "last_month",
            "first_count",
            "last_count",
            "first_share",
            "last_share",
            "share_delta",
            "trend",
        ]
    ].sort_values("share_delta", ascending=False, ignore_index=True)


def _plot_bar(df: pd.DataFrame, x: str, y: str, path: Path, title: str, color: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df[x].astype(str), df[y], color=color)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(y.replace("_", " ").title())
    ax.tick_params(axis="x", rotation=45)
    if len(df) > 14:
        step = max(1, math.ceil(len(df) / 12))
        ticks = list(range(0, len(df), step))
        if ticks[-1] != len(df) - 1:
            ticks.append(len(df) - 1)
        labels = df[x].astype(str).tolist()
        ax.set_xticks(ticks)
        ax.set_xticklabels([labels[i] for i in ticks])
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def prepare_support_aware_drift_plot_data(
    drift: pd.DataFrame,
    support_column: str = "jobs_in_month",
    low_support_threshold: int = 100,
) -> pd.DataFrame:
    """Return centroid-drift rows with explicit support flags for plotting/tests."""
    if drift.empty:
        return pd.DataFrame(columns=["month", "centroid_drift", "n_jobs", "low_support", "marker_size"])
    out = drift.dropna(subset=["centroid_drift"]).copy().reset_index(drop=True)
    if support_column in out.columns:
        out["n_jobs"] = pd.to_numeric(out[support_column], errors="coerce").fillna(0).astype(int)
    else:
        out["n_jobs"] = 0
    out["low_support"] = out["n_jobs"] < low_support_threshold
    max_n = max(float(out["n_jobs"].max()), 1.0)
    out["marker_size"] = 35.0 + 185.0 * np.sqrt(out["n_jobs"].clip(lower=0) / max_n)
    return out[["month", "centroid_drift", "n_jobs", "low_support", "marker_size"]]


def _plot_support_aware_drift(
    drift: pd.DataFrame,
    path: Path,
    title: str,
    support_column: str = "jobs_in_month",
    low_support_threshold: int = 100,
    salary_subset: bool = False,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_df = prepare_support_aware_drift_plot_data(drift, support_column, low_support_threshold)
    fig, ax = plt.subplots(figsize=(10, 5.6))
    if plot_df.empty:
        ax.text(0.5, 0.5, "No hay deriva temporal suficiente para graficar", ha="center", va="center")
        ax.axis("off")
    else:
        x = np.arange(len(plot_df))
        reliable = ~plot_df["low_support"]
        ax.plot(x, plot_df["centroid_drift"], color="#64748b", linewidth=1.2, alpha=0.7)
        ax.vlines(x, 0, plot_df["centroid_drift"], color="#cbd5e1", linewidth=1.0)
        ax.scatter(
            x[reliable],
            plot_df.loc[reliable, "centroid_drift"],
            s=plot_df.loc[reliable, "marker_size"],
            color="#5b6ee1",
            edgecolor="white",
            linewidth=0.8,
            label=f"n >= {low_support_threshold}",
            zorder=3,
        )
        ax.scatter(
            x[~reliable],
            plot_df.loc[~reliable, "centroid_drift"],
            s=plot_df.loc[~reliable, "marker_size"],
            color="#cbd5e1",
            edgecolor="#64748b",
            linewidth=0.8,
            label=f"n < {low_support_threshold}",
            zorder=3,
        )
        for idx, row in plot_df[plot_df["low_support"]].iterrows():
            ax.annotate(
                f"n={int(row['n_jobs'])}",
                (int(idx), float(row["centroid_drift"])),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=7,
                color="#475569",
            )
        labels = plot_df["month"].astype(str).tolist()
        step = max(1, math.ceil(len(labels) / 12))
        ticks = list(range(0, len(labels), step))
        if ticks[-1] != len(labels) - 1:
            ticks.append(len(labels) - 1)
        ax.set_xticks(ticks)
        ax.set_xticklabels([labels[i] for i in ticks], rotation=45, ha="right")
        ax.set_ylabel("Distancia coseno")
        ax.set_xlabel("Intervalo")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=18)
        subtitle = "Distancia coseno entre centroides semánticos consecutivos"
        if salary_subset:
            subtitle += "; subset USD con salario utilizable"
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color="#475569")
        ax.text(
            0.99,
            0.02,
            "Bins con bajo n son diagnósticos, no tendencia estable.",
            transform=ax.transAxes,
            ha="right",
            fontsize=8,
            color="#64748b",
        )
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def prepare_salary_coverage_plot_data(
    salary_metadata: pd.DataFrame,
    low_support_threshold: int = 100,
) -> tuple[pd.DataFrame, float]:
    """Return salary coverage rows with overall reference and low-n flags."""
    if salary_metadata.empty:
        return pd.DataFrame(columns=["month", "n_jobs", "n_salary_jobs", "salary_coverage", "low_support"]), 0.0
    out = salary_metadata.copy().reset_index(drop=True)
    out["n_jobs"] = pd.to_numeric(out.get("n_jobs", 0), errors="coerce").fillna(0).astype(int)
    out["n_salary_jobs"] = pd.to_numeric(out.get("n_salary_jobs", 0), errors="coerce").fillna(0).astype(int)
    if "salary_coverage" not in out:
        out["salary_coverage"] = np.where(out["n_jobs"] > 0, out["n_salary_jobs"] / out["n_jobs"], 0.0)
    out["salary_coverage"] = pd.to_numeric(out["salary_coverage"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    out["low_support"] = out["n_jobs"] < low_support_threshold
    total_jobs = int(out["n_jobs"].sum())
    overall = float(out["n_salary_jobs"].sum() / total_jobs) if total_jobs else 0.0
    return out[["month", "n_jobs", "n_salary_jobs", "salary_coverage", "low_support"]], overall


def _plot_salary_coverage(
    salary_metadata: pd.DataFrame,
    path: Path,
    low_support_threshold: int = 100,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    plot_df, overall = prepare_salary_coverage_plot_data(salary_metadata, low_support_threshold)
    fig, ax = plt.subplots(figsize=(10, 5.6))
    if plot_df.empty:
        ax.text(0.5, 0.5, "No hay metadatos salariales suficientes", ha="center", va="center")
        ax.axis("off")
    else:
        x = np.arange(len(plot_df))
        colors = np.where(plot_df["low_support"], "#cbd5e1", "#2563eb")
        ax.bar(x, plot_df["salary_coverage"], color=colors, edgecolor="#64748b", linewidth=0.4)
        ax.axhline(overall, color="#ef4444", linestyle="--", linewidth=1.2, label=f"Cobertura total: {overall:.0%}")
        for idx, row in plot_df.iterrows():
            if bool(row["low_support"]) or idx % max(1, math.ceil(len(plot_df) / 14)) == 0:
                ax.annotate(
                    f"n={int(row['n_jobs'])}",
                    (idx, float(row["salary_coverage"])),
                    textcoords="offset points",
                    xytext=(0, 4),
                    ha="center",
                    fontsize=7,
                    color="#475569",
                    rotation=90 if len(plot_df) > 25 else 0,
                )
        labels = plot_df["month"].astype(str).tolist()
        step = max(1, math.ceil(len(labels) / 12))
        ticks = list(range(0, len(labels), step))
        if ticks[-1] != len(labels) - 1:
            ticks.append(len(labels) - 1)
        ax.set_xticks(ticks)
        ax.set_xticklabels([labels[i] for i in ticks], rotation=45, ha="right")
        ax.set_ylim(0, min(1.05, max(1.0, float(plot_df["salary_coverage"].max()) + 0.08)))
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_ylabel("Cobertura salarial")
        ax.set_xlabel("Intervalo")
        ax.set_title("Cobertura salarial por intervalo", fontsize=13, fontweight="bold", pad=18)
        ax.text(
            0.0,
            1.02,
            "Postings con salario utilizable / postings totales del intervalo",
            transform=ax.transAxes,
            fontsize=9,
            color="#475569",
        )
        ax.text(
            0.99,
            0.02,
            "Campos salariales incompletos: no es un censo salarial del mercado.",
            transform=ax.transAxes,
            ha="right",
            fontsize=8,
            color="#64748b",
        )
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _write_required_plots(
    sampled: pd.DataFrame,
    drift: pd.DataFrame,
    growth: pd.DataFrame,
    figures_dir: Path,
) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        figures_dir / "job_volume_by_week.png",
        figures_dir / "centroid_drift_by_week.png",
        figures_dir / "top_rising_skills.png",
        figures_dir / "top_declining_skills.png",
    ]
    volume = sampled.groupby("month").size().rename("jobs").reset_index()
    time_bin = str(sampled.get("_time_bin", pd.Series(["M"])).dropna().iloc[0]) if "_time_bin" in sampled else "M"
    period_label = {"D": "Day", "H": "Hour", "W": "Week", "M": "Month"}.get(time_bin.upper(), "Period")
    spanish_period = {"Day": "día", "Hour": "hora", "Week": "semana", "Month": "mes", "Period": "intervalo"}.get(period_label, "intervalo")
    _plot_bar(volume, "month", "jobs", paths[0], f"Volumen de postings por {spanish_period}", "#2f6f73")
    _plot_support_aware_drift(
        drift.dropna(subset=["centroid_drift"]),
        paths[1],
        "Desplazamiento semántico entre intervalos",
    )
    rising = growth.sort_values("share_delta", ascending=False).head(15)
    declining = growth[growth["share_delta"] < 0].sort_values("share_delta", ascending=True).head(15).copy()
    declining["share_delta_abs"] = declining["share_delta"].abs()
    _plot_bar(rising, "skill", "share_delta", paths[2], "Skills con mayor aumento en la ventana", "#1b9e77")
    _plot_bar(declining, "skill", "share_delta_abs", paths[3], "Skills con mayor caída en la ventana", "#d95f02")
    return paths


def _write_salary_weighted_plots(
    salary_drift: pd.DataFrame,
    salary_metadata: pd.DataFrame,
    figures_dir: Path,
    period_label: str = "Period",
) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if not salary_drift.empty:
        drift_plot = figures_dir / "centroid_drift_salary_weighted_by_week.png"
        plot_df = salary_drift[["month_to", "cosine_distance"]].rename(
            columns={"month_to": "month", "cosine_distance": "centroid_drift"}
        )
        _plot_support_aware_drift(
            plot_df,
            drift_plot,
            "Desplazamiento semántico ponderado por salario",
            salary_subset=True,
        )
        paths.append(drift_plot)
    if not salary_metadata.empty:
        coverage_plot = figures_dir / "salary_coverage_by_week.png"
        _plot_salary_coverage(salary_metadata, coverage_plot)
        paths.append(coverage_plot)
    return paths


def _optional_cluster_outputs(
    sampled: pd.DataFrame,
    vectors: Any,
    output_dir: Path,
    figures_dir: Path,
) -> tuple[list[Path], list[dict[str, Any]], str | None]:
    try:
        dense = vectors.toarray() if sparse.issparse(vectors) else np.asarray(vectors)
        if dense.shape[0] < 8 or dense.shape[1] < 2:
            return [], [], "Skipped: fewer than 8 rows or fewer than 2 vector dimensions."
        n_distinct = np.unique(np.round(dense, decimals=12), axis=0).shape[0]
        if n_distinct < 2:
            return [], [], "Skipped: fewer than 2 distinct vector points."
        n_clusters = min(8, dense.shape[0], n_distinct)
        labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(dense)
        cluster_df = sampled[["job_id", "title", "month"]].copy()
        cluster_df["cluster_id"] = labels
        cluster_path = output_dir / "job_clusters.parquet"
        cluster_df.to_parquet(cluster_path, index=False)

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_path = figures_dir / "job_cluster_map_svd.png"
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(dense[:, 0], dense[:, 1], c=labels, cmap="tab10", s=18, alpha=0.75)
        ax.set_title("Mapa diagnóstico de clusters de postings")
        ax.set_xlabel("Componente 1")
        ax.set_ylabel("Componente 2")
        fig.colorbar(scatter, ax=ax, label="Cluster diagnóstico")
        fig.tight_layout()
        fig.savefig(fig_path, dpi=140)
        plt.close(fig)

        summary = []
        for cluster_id, group in cluster_df.groupby("cluster_id"):
            titles = group["title"].fillna("").astype(str).value_counts().head(5).index.tolist()
            summary.append(
                {
                    "cluster_id": int(cluster_id),
                    "n_jobs": int(len(group)),
                    "top_titles": titles,
                }
            )
        return [cluster_path, fig_path], summary, None
    except Exception as exc:
        return [], [], f"Skipped: {exc}"


def _optional_similarity_outputs(vectors: Any, output_dir: Path, figures_dir: Path) -> tuple[list[Path], dict[str, Any] | None, str | None]:
    try:
        n = min(500, vectors.shape[0])
        if n < 2:
            return [], None, "Skipped: fewer than 2 sampled rows."
        subset = vectors[:n]
        sims = cosine_similarity(subset)
        values = sims[~np.eye(n, dtype=bool)]
        sample_path = output_dir / "similarity_sample.parquet"
        pd.DataFrame({"cosine_similarity": values}).to_parquet(sample_path, index=False)

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_path = figures_dir / "similarity_distribution.png"
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(values, bins=30, color="#4c78a8")
        ax.set_title("Sample Similarity Distribution")
        ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("Pair count")
        fig.tight_layout()
        fig.savefig(fig_path, dpi=140)
        plt.close(fig)
        summary = {
            "n_jobs": int(n),
            "n_pairs": int(len(values)),
            "mean_similarity": float(np.mean(values)),
            "median_similarity": float(np.median(values)),
        }
        return [sample_path, fig_path], summary, None
    except Exception as exc:
        return [], None, f"Skipped: {exc}"


def _format_table(df: pd.DataFrame, columns: list[str], limit: int) -> str:
    if df.empty:
        return "_No rows available._"
    table = df[columns].head(limit).copy()
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in table.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join(rows)


def _write_report(
    report_path: Path,
    command_used: str,
    manifest: dict[str, Any],
    drift: pd.DataFrame,
    growth: pd.DataFrame,
    cluster_summary: list[dict[str, Any]],
    cluster_note: str | None,
    similarity_summary: dict[str, Any] | None,
    similarity_note: str | None,
    comparison_note: str | None = None,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    largest_drifts = drift.dropna(subset=["centroid_drift"]).sort_values("centroid_drift", ascending=False)
    rising = growth.sort_values("share_delta", ascending=False)
    declining = growth[growth["share_delta"] < 0].sort_values("share_delta", ascending=True)

    lines = [
        "# M2.3 Temporal Drift Demo",
        "",
        "## Command",
        f"`{command_used}`",
        "",
        "## Run Summary",
        f"- Input path: `{manifest['input_path']}`",
        f"- Rows input: {manifest['n_rows_input']}",
        f"- Rows selected: {manifest['n_rows_selected']}",
        f"- listed_time parse success rate: {manifest['listed_time_parse_success_rate']:.4f}",
        f"- Time column: `{manifest.get('time_column', 'listed_time')}` ({manifest.get('time_column_interpretation', 'canonical posting-listing time')})",
        f"- Sampling mode: {manifest['sampling_mode']}",
        f"- Representation: `{manifest.get('representation', manifest.get('embedding_method', 'tfidf_svd'))}`",
        f"- Centroid weighting: `{manifest.get('centroid_weighting', 'unweighted')}`",
        f"- Embedding backend: `{manifest.get('embedding_backend')}`",
        f"- Embedding model: `{manifest.get('embedding_model')}`",
        f"- Month range: {manifest['first_month']} to {manifest['last_month']}",
        f"- Months covered: {manifest['n_months']}",
        f"- Reliability label: `{manifest.get('reliability_label', 'demo_only')}`",
        f"- Runtime seconds: {manifest['runtime_seconds']:.2f}",
        "",
        "## Reliability Gates",
    ]
    warnings = manifest.get("reliability_warnings", [])
    lines.extend([f"- {warning}" for warning in warnings] if warnings else ["- No reliability warnings."])
    monthly_counts = pd.DataFrame(
        [{"month": month, "rows": count} for month, count in manifest.get("monthly_row_counts", {}).items()]
    )
    lines.extend(
        [
            "",
            "## Monthly Row Counts",
            _format_table(monthly_counts, ["month", "rows"], 100),
            "",
            "## Top 10 Largest Centroid Drifts",
            _format_table(largest_drifts, ["month", "previous_month", "centroid_drift", "jobs_in_month"], 10),
            "",
            "## Top 15 Rising Skills",
            _format_table(rising, ["skill", "first_share", "last_share", "share_delta"], 15),
            "",
            "## Top 15 Declining Skills",
            _format_table(declining, ["skill", "first_share", "last_share", "share_delta"], 15),
            "",
            "## Figures",
        ]
    )
    lines.extend(f"- `{path}`" for path in manifest["generated_files"] if str(path).endswith(".png"))
    if cluster_summary:
        lines.extend(["", "## Optional Clustering Summary"])
        lines.append(_format_table(pd.DataFrame(cluster_summary), ["cluster_id", "n_jobs", "top_titles"], 50))
    elif cluster_note:
        lines.extend(["", "## Optional Clustering Summary", cluster_note])
    if similarity_summary:
        lines.extend(
            [
                "",
                "## Optional Similarity Summary",
                f"- Jobs sampled: {similarity_summary['n_jobs']}",
                f"- Pair similarities: {similarity_summary['n_pairs']}",
                f"- Mean similarity: {similarity_summary['mean_similarity']:.4f}",
                f"- Median similarity: {similarity_summary['median_similarity']:.4f}",
            ]
        )
    elif similarity_note:
        lines.extend(["", "## Optional Similarity Summary", similarity_note])
    if comparison_note:
        lines.extend(["", "## Comparison", comparison_note])
    if manifest.get("centroid_weighting") in {"salary", "both"}:
        lines.extend(
            [
                "",
                "## Salary-Weighted Centroid View",
                "- Salary-weighted centroids describe the salary-disclosed USD subset, not the full job market.",
                f"- Salary rows used: {manifest.get('salary_rows_used', 0)}",
                f"- Salary coverage in selected rows: {manifest.get('salary_coverage', 0.0):.4f}",
                f"- Salary weighting strategy: `{manifest.get('salary_weight_strategy')}`",
                f"- Salary-weighted drift path: `{manifest.get('salary_weighted_drift_path')}`",
            ]
        )
    next_step = "Run temporal audit first, then use larger samples only if local memory allows."
    if manifest["sample_size"] >= 100000:
        next_step = "Create the meeting summary, then evaluate a small explicit semantic embedding smoke run."
    lines.extend(
        [
            "",
            "## Known Limitations",
            "- This is a temporal comparison / temporal drift demo, not evidence of market evolution unless coverage is sufficient.",
            "- TF-IDF/SVD is a fast baseline, not a semantic embedding model.",
            "- Semantic embedding runs should stay small on 4 GB VRAM / 8 GB RAM machines unless explicitly validated.",
            "- Numeric `listed_time` values are interpreted as epoch milliseconds or seconds when plain pandas parsing lands before 1990.",
            "- Skill growth depends on the quality and consistency of `skills_text`.",
            "- Month-to-month drift can be noisy when a month has few postings.",
            "- FAISS is not needed for this milestone because the demo computes aggregate centroids, not large-scale ANN retrieval.",
            "",
            "## Deferred Tasks",
            "- Bootstrap stability for drift and skill growth is deferred; add a small explicit seed list before using confidence intervals.",
            "",
            "## Next Step",
            next_step,
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _comparison_note(output_dir: Path, manifest: dict[str, Any], drift: pd.DataFrame) -> str | None:
    if output_dir.name != "trends_100k":
        return None
    ten_k_dir = output_dir.parent / "trends_10k"
    ten_k_manifest_path = ten_k_dir / "temporal_manifest.json"
    ten_k_drift_path = ten_k_dir / "weekly_centroid_drift.parquet"
    if not ten_k_manifest_path.exists() or not ten_k_drift_path.exists():
        return None
    try:
        ten_k_manifest = json.loads(ten_k_manifest_path.read_text(encoding="utf-8"))
        ten_k_drift = pd.read_parquet(ten_k_drift_path)
        current_drift = drift["centroid_drift"].dropna().max()
        previous_drift = ten_k_drift["centroid_drift"].dropna().max()
        return (
            f"- 10k selected rows: {ten_k_manifest['n_rows_selected']} across "
            f"{ten_k_manifest['first_month']} to {ten_k_manifest['last_month']}.\n"
            f"- 100k selected rows: {manifest['n_rows_selected']} across "
            f"{manifest['first_month']} to {manifest['last_month']}.\n"
            f"- Largest centroid drift changed from {previous_drift:.4f} at 10k "
            f"to {current_drift:.4f} at 100k."
        )
    except Exception:
        return None


def run_temporal_demo(
    silver_path: Path,
    output_dir: Path,
    figures_dir: Path,
    report_path: Path,
    sample_size: int = 10000,
    sampling_mode: str = "temporal-stride",
    command_used: str | None = None,
    min_jobs_per_month: int = 1,
    time_bin: str = "M",
    representation: str = "tfidf_svd",
    embedding_backend: str = "mock",
    embedding_model: str = "deterministic-mock",
    embedding_batch_size: int = 16,
    embedding_cache_dir: Path | None = None,
    device: str = "cpu",
    max_embedding_rows: int | None = None,
    time_column: str = "listed_time",
    centroid_weighting: str = "unweighted",
) -> TemporalDemoResult:
    """Run the temporal analytics demo and write all required outputs."""
    started_at = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    if representation not in {"tfidf_svd", "semantic_embeddings"}:
        raise ValueError(f"Unsupported representation: {representation}")
    if centroid_weighting not in {"unweighted", "salary", "both"}:
        raise ValueError(f"Unsupported centroid_weighting: {centroid_weighting}")
    if embedding_batch_size <= 0:
        raise ValueError("embedding_batch_size must be positive.")
    df = pd.read_parquet(silver_path)
    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"Input data must contain '{TEXT_COLUMN}'.")
    if SKILL_COLUMN not in df.columns:
        raise ValueError(f"Input data must contain '{SKILL_COLUMN}'.")

    parsed = parse_time_column(df, time_column)
    n_valid_time = int(parsed.notna().sum())
    parse_rate = float(n_valid_time / len(df)) if len(df) else 0.0
    sampled = sample_jobs(
        df,
        sample_size=sample_size,
        sampling_mode=sampling_mode,
        time_column=time_column,
        time_bin=time_bin,
    )
    temporal = sampled.dropna(subset=["_listed_time_parsed"]).copy()
    temporal = temporal[temporal.groupby("month")["month"].transform("size") >= min_jobs_per_month]

    semantic_limit_applied = False
    if representation == "semantic_embeddings" and max_embedding_rows is not None and len(temporal) > max_embedding_rows:
        semantic_limit_applied = True
        temporal = sample_jobs(
            temporal.drop(columns=["_listed_time_parsed"], errors="ignore"),
            sample_size=max_embedding_rows,
            sampling_mode=sampling_mode,
            time_column=time_column,
            time_bin=time_bin,
        )
        temporal = temporal.dropna(subset=["_listed_time_parsed"]).copy()

    centroid_metadata_path: Path | None = None
    centroid_storage_path: Path | None = None
    embedding_provenance: dict[str, Any] = {}
    if representation == "tfidf_svd":
        vectors, embedding_method, svd_enabled = _vectorize_texts(temporal[TEXT_COLUMN])
        drift = compute_centroid_drift(temporal, vectors)
    else:
        cache_dir = embedding_cache_dir or Path("data/cache/embeddings")
        vectors, embedding_provenance = _load_or_compute_embeddings(
            temporal[TEXT_COLUMN],
            backend_name=embedding_backend,
            model_name=embedding_model,
            batch_size=embedding_batch_size,
            device=device,
            cache_dir=cache_dir,
        )
        centroid_storage_path = output_dir / "weekly_centroids.npy"
        centroid_metadata_path = output_dir / "weekly_centroid_metadata.parquet"
        drift, centroid_metadata = compute_semantic_centroid_drift(
            temporal,
            vectors,
            embedding_backend=embedding_backend,
            embedding_model=embedding_model,
            centroids_path=centroid_storage_path,
        )
        centroid_metadata.to_parquet(centroid_metadata_path, index=False)
        embedding_method = "semantic_embeddings"
        svd_enabled = False
    growth = compute_skill_growth(temporal)

    weekly_drift_path = output_dir / "weekly_centroid_drift.parquet"
    skill_growth_path = output_dir / "skill_growth.parquet"
    manifest_path = output_dir / "temporal_manifest.json"
    drift.to_parquet(weekly_drift_path, index=False)
    growth.to_parquet(skill_growth_path, index=False)
    generated_paths = [weekly_drift_path, skill_growth_path, manifest_path]
    if centroid_metadata_path is not None:
        generated_paths.append(centroid_metadata_path)
    if centroid_storage_path is not None:
        generated_paths.append(centroid_storage_path)

    salary_summary: dict[str, Any] = {}
    salary_weighted_drift_path: Path | None = None
    salary_weighted_metadata_path: Path | None = None
    salary_weight_diagnostics_path: Path | None = None
    salary_weighted_centroids_path: Path | None = None
    salary_plot_paths: list[Path] = []
    if centroid_weighting in {"salary", "both"}:
        salary_drift, salary_metadata, salary_diagnostics, salary_summary, salary_weighted_centroids_path = (
            compute_salary_weighted_centroid_drift(
                temporal,
                vectors,
                representation=representation,
                time_column=time_column,
                output_dir=output_dir,
                embedding_backend=embedding_backend if representation == "semantic_embeddings" else None,
                embedding_model=embedding_model if representation == "semantic_embeddings" else None,
            )
        )
        salary_weighted_drift_path = output_dir / "weekly_centroid_drift_salary_weighted.parquet"
        salary_weighted_metadata_path = output_dir / "weekly_centroid_metadata_salary_weighted.parquet"
        salary_weight_diagnostics_path = output_dir / "salary_weight_diagnostics.parquet"
        salary_drift.to_parquet(salary_weighted_drift_path, index=False)
        salary_metadata.to_parquet(salary_weighted_metadata_path, index=False)
        salary_diagnostics.to_parquet(salary_weight_diagnostics_path, index=False)
        period_label = {"D": "Day", "H": "Hour", "W": "Week", "M": "Month"}.get(time_bin.upper(), "Period")
        salary_plot_paths = _write_salary_weighted_plots(
            salary_drift,
            salary_metadata,
            figures_dir,
            period_label=period_label,
        )
        generated_paths.extend(
            [
                salary_weighted_drift_path,
                salary_weighted_metadata_path,
                salary_weight_diagnostics_path,
                salary_weighted_centroids_path,
                *salary_plot_paths,
            ]
        )

    generated_paths.extend(_write_required_plots(temporal, drift, growth, figures_dir))

    cluster_paths, cluster_summary, cluster_note = _optional_cluster_outputs(temporal, vectors, output_dir, figures_dir)
    similarity_paths, similarity_summary, similarity_note = _optional_similarity_outputs(vectors, output_dir, figures_dir)
    generated_paths.extend(cluster_paths)
    generated_paths.extend(similarity_paths)

    weeks = sorted(temporal["month"].dropna().unique().tolist())
    weekly_row_counts = {str(week): int(count) for week, count in temporal.groupby("month").size().to_dict().items()}
    reliability = build_reliability_assessment(weekly_row_counts)
    reliability_warnings = list(reliability["warnings"])
    known_limitations = list(reliability["known_limitations"])
    if semantic_limit_applied:
        reliability_warnings.append(
            f"Semantic embedding rows were capped at {max_embedding_rows}; results are a capped smoke run."
        )
        known_limitations.append("Semantic embedding mode is capped by default for small local machines.")
    manifest = {
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "input_path": str(silver_path),
        "output_dir": str(output_dir),
        "figures_dir": str(figures_dir),
        "report_path": str(report_path),
        "n_rows_input": int(len(df)),
        "n_rows_valid_time": n_valid_time,
        "valid_time_rows": n_valid_time,
        "n_rows_selected": int(len(temporal)),
        "listed_time_parse_success_rate": parse_rate,
        "time_parse_success_rate": parse_rate,
        "sample_size": int(sample_size),
        "sampling_mode": sampling_mode,
        "time_column": time_column,
        "time_bin": time_bin.upper(),
        "time_column_interpretation": TIME_COLUMN_INTERPRETATIONS.get(time_column, "temporal column"),
        "centroid_weighting": centroid_weighting,
        "min_jobs_per_month": int(min_jobs_per_month),
        "first_month": weeks[0] if weeks else None,
        "last_month": weeks[-1] if weeks else None,
        "n_weeks": int(len(weeks)),
        "n_months": int(len(weeks)),
        "runtime_seconds": float(time.perf_counter() - started_at),
        "text_column": TEXT_COLUMN,
        "skill_column": SKILL_COLUMN,
        "representation": representation,
        "embedding_backend": embedding_backend if representation == "semantic_embeddings" else None,
        "embedding_model": embedding_model if representation == "semantic_embeddings" else None,
        "embedding_batch_size": int(embedding_batch_size) if representation == "semantic_embeddings" else None,
        "embedding_cache_dir": str(embedding_cache_dir or Path("data/cache/embeddings"))
        if representation == "semantic_embeddings"
        else None,
        "embedding_cache_path": embedding_provenance.get("embedding_cache_path"),
        "embedding_cache_hit": embedding_provenance.get("embedding_cache_hit"),
        "device": device if representation == "semantic_embeddings" else None,
        "max_embedding_rows": max_embedding_rows if representation == "semantic_embeddings" else None,
        "semantic_limit_applied": bool(semantic_limit_applied),
        "weekly_row_counts": weekly_row_counts,
        "monthly_row_counts": weekly_row_counts,
        "reliability_label": reliability["label"],
        "reliability_warnings": reliability_warnings,
        "known_limitations": known_limitations,
        "salary_weighted_drift_path": str(salary_weighted_drift_path) if salary_weighted_drift_path else None,
        "salary_weighted_metadata_path": str(salary_weighted_metadata_path)
        if salary_weighted_metadata_path
        else None,
        "salary_weight_diagnostics_path": str(salary_weight_diagnostics_path)
        if salary_weight_diagnostics_path
        else None,
        "salary_weighted_centroids_path": str(salary_weighted_centroids_path)
        if salary_weighted_centroids_path
        else None,
        "weighted_centroid_warning": (
            "Salary-weighted centroids describe the salary-disclosed USD subset, not the full job market."
            if centroid_weighting in {"salary", "both"}
            else None
        ),
        "deferred_tasks": [
            "Bootstrap stability for drift and skill growth is deferred; add a small explicit seed list before using confidence intervals."
        ],
        "embedding_method": embedding_method,
        "svd_enabled": bool(svd_enabled),
        "generated_files": [str(path) for path in generated_paths],
    }
    manifest.update(embedding_provenance)
    manifest.update(salary_summary)
    _write_report(
        report_path=report_path,
        command_used=command_used or "python -m jobsrec.cli temporal-demo",
        manifest=manifest,
        drift=drift,
        growth=growth,
        cluster_summary=cluster_summary,
        cluster_note=cluster_note,
        similarity_summary=similarity_summary,
        similarity_note=similarity_note,
        comparison_note=_comparison_note(output_dir, manifest, drift),
    )
    generated_paths.append(report_path)
    manifest["generated_files"] = [str(path) for path in generated_paths]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return TemporalDemoResult(
        weekly_drift_path=weekly_drift_path,
        skill_growth_path=skill_growth_path,
        manifest_path=manifest_path,
        report_path=report_path,
        generated_files=[str(path) for path in generated_paths],
        manifest=manifest,
    )
