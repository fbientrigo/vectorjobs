"""Local skill-share evolution analytics for job postings.

This is a CPU-only, offline counterpart to
``notebooks/cluster_skill_timelines_colab.ipynb``: it assigns each posting to
one of five fixed domains and plots how the share of postings requiring each
skill changes over time, per domain.

Differences from the Colab notebook:
- Domain assignment uses TF-IDF + SVD job embeddings matched against domain
  description embeddings (cosine similarity), with a keyword fallback,
  instead of Qwen sentence embeddings.
- There is no LLM enrichment step. Skills come only from ``skills_text``.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from jobsrec.trends.temporal import build_reliability_assessment
from jobsrec.trends.temporal_clusters import (
    _format_table,
    _git_commit,
    _join_text_columns,
    _package_versions,
    _parse_datetime,
    _safe_strings,
    _select_rows,
    _time_bins,
    build_temporal_cluster_embeddings,
    detect_temporal_cluster_schema,
)


TARGET_DOMAINS = ["health", "tech", "construction_industry", "education", "sales"]

DOMAIN_LABELS = {
    "health": "Health",
    "tech": "Tech",
    "construction_industry": "Construction/Industry",
    "education": "Education",
    "sales": "Sales",
}

DOMAIN_CONFIG = {
    "health": {
        "keywords": ["patient", "nursing", "nurse", "medical", "clinic", "health care", "healthcare", "health", "hospital", "pharmacy", "physician", "therapy"],
        "prompt": "Jobs in health care, hospitals, clinics, nursing, patient care, pharmacy, medical operations, therapy, public health, and clinical support.",
    },
    "tech": {
        "keywords": ["software", "developer", "engineer", "information technology", "technology", "data", "python", "sql", "cloud", "machine learning", "cybersecurity", "devops"],
        "prompt": "Jobs in software engineering, data science, analytics, cloud infrastructure, cybersecurity, IT, product engineering, and developer tools.",
    },
    "construction_industry": {
        "keywords": ["construction", "manufacturing", "industrial", "warehouse", "logistics", "maintenance", "machinery", "site", "contractor", "safety", "electrical", "mechanical", "plant", "operator"],
        "prompt": "Jobs in construction, industrial operations, manufacturing, logistics, maintenance, plant operations, trades, site safety, and engineering operations.",
    },
    "education": {
        "keywords": ["education", "students", "student", "teacher", "teaching", "school", "training", "curriculum", "classroom", "academic", "instructor", "learning"],
        "prompt": "Jobs in education, schools, universities, teaching, curriculum, learning programs, academic administration, training, and student services.",
    },
    "sales": {
        "keywords": ["sales", "business development", "customer service", "account", "revenue", "client", "crm", "lead generation", "retail", "territory", "quota"],
        "prompt": "Jobs in sales, business development, account management, retail sales, customer success, revenue operations, client growth, and CRM workflows.",
    },
}

_CANONICAL_SKILLS = {
    "sql": "SQL",
    "python": "Python",
    "java": "Java",
    "javascript": "JavaScript",
    "aws": "AWS",
    "gcp": "GCP",
    "azure": "Azure",
    "c++": "C++",
    "c#": "C#",
    "r": "R",
}

DOMAIN_ASSIGNMENT_COLUMNS = [
    "job_id",
    "time_bin",
    "domain",
    "domain_label",
    "domain_score",
    "domain_margin",
    "embedding_domain",
    "keyword_domain",
    "keyword_score",
    "domain_source",
    "is_uncertain",
]

DOMAIN_SKILL_MONTHLY_COLUMNS = ["domain", "time_bin", "skill", "skill_job_count", "job_count", "share_pct"]


@dataclass(frozen=True)
class SkillEvolutionResult:
    """Paths and manifest for a skill-evolution run."""

    output_dir: Path
    manifest_path: Path
    report_path: Path
    domain_skill_monthly_path: Path
    generated_files: list[str]
    manifest: dict[str, Any]


def normalize_skill(value: object) -> str | None:
    """Normalize a raw skill string to a display-friendly canonical form."""
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .;:,|/\\")
    if not text:
        return None
    lowered = text.lower()
    if lowered in _CANONICAL_SKILLS:
        return _CANONICAL_SKILLS[lowered]
    return " ".join(part.capitalize() for part in lowered.split())


def split_skills(value: object) -> list[str]:
    """Split a delimited skills string into normalized, de-duplicated skills."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    parts = re.split(r"[;,|]", str(value))
    normalized: list[str] = []
    seen: set[str] = set()
    for part in parts:
        skill = normalize_skill(part)
        if skill and skill.lower() not in seen:
            normalized.append(skill)
            seen.add(skill.lower())
    return normalized


def _embed_domain_prompts(prompts: list[str], vectorizer: TfidfVectorizer, svd: Any | None) -> np.ndarray:
    tfidf = vectorizer.transform(prompts)
    embeddings = svd.transform(tfidf) if svd is not None else tfidf.toarray()
    return normalize(np.asarray(embeddings, dtype=np.float32), copy=False)


def _keyword_count(text: str, keywords: list[str]) -> int:
    normalized = " ".join(str(text).lower().replace("/", " ").replace("|", " ").replace(";", " ").split())
    return sum(normalized.count(keyword) for keyword in keywords)


def assign_job_domains(
    df: pd.DataFrame,
    job_embeddings: np.ndarray,
    vectorizer: TfidfVectorizer,
    svd: Any | None,
    text_column: str,
    skill_column: str | None,
    confidence_threshold: float,
    margin_threshold: float,
) -> pd.DataFrame:
    """Assign each row to one of TARGET_DOMAINS using TF-IDF similarity plus keyword fallback."""
    seed_prompts = [DOMAIN_CONFIG[domain]["prompt"] for domain in TARGET_DOMAINS]
    seed_embeddings = _embed_domain_prompts(seed_prompts, vectorizer, svd)

    similarities = job_embeddings @ seed_embeddings.T
    best_idx = similarities.argmax(axis=1)
    sorted_sims = np.sort(similarities, axis=1)
    best_scores = sorted_sims[:, -1]
    second_scores = sorted_sims[:, -2] if similarities.shape[1] > 1 else np.zeros(len(similarities))
    margins = best_scores - second_scores

    combined_text = _safe_strings(df[text_column])
    if skill_column and skill_column in df.columns:
        combined_text = combined_text.str.cat(_safe_strings(df[skill_column]), sep=" ")

    rows: list[dict[str, Any]] = []
    for i in range(len(df)):
        text = combined_text.iat[i]
        embedding_domain = TARGET_DOMAINS[int(best_idx[i])]
        keyword_scores = {domain: _keyword_count(text, DOMAIN_CONFIG[domain]["keywords"]) for domain in TARGET_DOMAINS}
        keyword_domain, keyword_score = max(keyword_scores.items(), key=lambda item: item[1])

        domain = embedding_domain
        source = "embedding"
        if keyword_score >= 2 or (keyword_score >= 1 and best_scores[i] < confidence_threshold):
            domain = keyword_domain
            source = "keyword_fallback"

        is_uncertain = bool(best_scores[i] < confidence_threshold or margins[i] < margin_threshold)
        rows.append(
            {
                "domain": domain,
                "domain_label": DOMAIN_LABELS[domain],
                "domain_score": float(best_scores[i]),
                "domain_margin": float(margins[i]),
                "embedding_domain": embedding_domain,
                "keyword_domain": keyword_domain if keyword_score > 0 else None,
                "keyword_score": int(keyword_score),
                "domain_source": source,
                "is_uncertain": is_uncertain,
            }
        )

    result = pd.DataFrame(rows)
    result["domain"] = pd.Categorical(result["domain"], categories=TARGET_DOMAINS)
    return result


def build_skill_long_table(df: pd.DataFrame, skill_column: str | None) -> pd.DataFrame:
    """Return one row per (job_id, normalized skill) from skills_text only."""
    if skill_column is None or skill_column not in df.columns:
        return pd.DataFrame(columns=["job_id", "skill"])
    rows: list[dict[str, Any]] = []
    for job_id, value in zip(df["job_id"], df[skill_column]):
        for skill in split_skills(value):
            rows.append({"job_id": job_id, "skill": skill})
    return pd.DataFrame(rows, columns=["job_id", "skill"])


def compute_domain_skill_monthly(job_skills_long: pd.DataFrame, job_domains: pd.DataFrame) -> pd.DataFrame:
    """Compute the share of each domain's postings that require each skill, per time bin."""
    domain_job_counts = (
        job_domains.groupby(["domain", "time_bin"], observed=False)["job_id"]
        .nunique()
        .reset_index(name="job_count")
    )
    if job_skills_long.empty:
        return pd.DataFrame(columns=DOMAIN_SKILL_MONTHLY_COLUMNS)

    skill_domain = job_skills_long.merge(job_domains[["job_id", "domain", "time_bin"]], on="job_id", how="left")
    skill_counts = (
        skill_domain.dropna(subset=["skill", "time_bin"])
        .groupby(["domain", "time_bin", "skill"], observed=True)["job_id"]
        .nunique()
        .reset_index(name="skill_job_count")
    )
    if skill_counts.empty:
        return pd.DataFrame(columns=DOMAIN_SKILL_MONTHLY_COLUMNS)

    domain_skill = skill_counts.merge(domain_job_counts, on=["domain", "time_bin"], how="left")
    domain_skill["share_pct"] = np.where(
        domain_skill["job_count"] > 0,
        100.0 * domain_skill["skill_job_count"] / domain_skill["job_count"],
        0.0,
    )
    domain_skill["share_pct"] = domain_skill["share_pct"].clip(lower=0, upper=100)
    return domain_skill[DOMAIN_SKILL_MONTHLY_COLUMNS]


def prepare_skill_composition_plot_data(
    domain_df: pd.DataFrame,
    top_n: int = 5,
    low_support_threshold: int = 100,
) -> tuple[pd.DataFrame, list[str]]:
    """Normalize skill mentions to a per-bin 100% composition with Other."""
    columns = ["time_bin", "skill", "skill_job_count", "composition_share", "job_count", "low_support"]
    if domain_df.empty:
        return pd.DataFrame(columns=columns), []
    work = domain_df.copy()
    work["skill_job_count"] = pd.to_numeric(work["skill_job_count"], errors="coerce").fillna(0.0)
    work["job_count"] = pd.to_numeric(work["job_count"], errors="coerce").fillna(0).astype(int)
    top_skills = (
        work.groupby("skill")["skill_job_count"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index
        .astype(str)
        .tolist()
    )
    top_skills = [skill for skill in top_skills if skill.lower() != "other"]
    rows: list[dict[str, object]] = []
    for time_bin, group in work.groupby("time_bin", sort=True):
        total_mentions = float(group["skill_job_count"].sum())
        if total_mentions <= 0:
            continue
        support = int(group["job_count"].max())
        for skill in top_skills:
            count = float(group.loc[group["skill"].astype(str) == skill, "skill_job_count"].sum())
            rows.append(
                {
                    "time_bin": str(time_bin),
                    "skill": skill,
                    "skill_job_count": count,
                    "composition_share": count / total_mentions,
                    "job_count": support,
                    "low_support": support < low_support_threshold,
                }
            )
        other_count = float(group.loc[~group["skill"].astype(str).isin(top_skills), "skill_job_count"].sum())
        rows.append(
            {
                "time_bin": str(time_bin),
                "skill": "Other",
                "skill_job_count": other_count,
                "composition_share": other_count / total_mentions,
                "job_count": support,
                "low_support": support < low_support_threshold,
            }
        )
    result = pd.DataFrame(rows, columns=columns)
    ordered_skills = top_skills + ["Other"] if top_skills else []
    return result, ordered_skills


def _write_skill_evolution_plot(domain_df: pd.DataFrame, domain_label: str, path: Path, top_n: int) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    if domain_df.empty or domain_df["time_bin"].nunique() == 0:
        ax.text(0.5, 0.5, f"No hay datos suficientes de skills para {domain_label}", ha="center", va="center")
        ax.axis("off")
        status = "insufficient_data"
    else:
        composition, ordered_skills = prepare_skill_composition_plot_data(domain_df, top_n=min(top_n, 5))
        if composition.empty:
            ax.text(0.5, 0.5, f"No hay menciones de skills para {domain_label}", ha="center", va="center")
            ax.axis("off")
            status = "insufficient_data"
        else:
            unique_bins = sorted(composition["time_bin"].dropna().unique())
            x = np.arange(len(unique_bins))
            matrix = []
            pivot = composition.pivot_table(
                index="time_bin",
                columns="skill",
                values="composition_share",
                aggfunc="sum",
                fill_value=0.0,
            )
            for skill in ordered_skills:
                values = pivot.reindex(unique_bins).get(skill, pd.Series(0.0, index=unique_bins)).to_numpy(dtype=float)
                matrix.append(values)
            colors = ["#2563eb", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#94a3b8"]
            ax.stackplot(x, matrix, labels=ordered_skills, colors=colors[: len(matrix)], alpha=0.9)
            support = composition.groupby("time_bin")["job_count"].max().reindex(unique_bins).fillna(0).astype(int)
            low_support = support < 100
            for idx, is_low in enumerate(low_support.tolist()):
                if is_low:
                    ax.axvspan(idx - 0.5, idx + 0.5, color="#e2e8f0", alpha=0.35, zorder=0)
            ax.set_title(f"Composición de skills en {domain_label}", fontsize=13, fontweight="bold", pad=18, color="#1e293b")
            ax.text(
                0.0,
                1.02,
                "Share de composición: top-5 + Other suma 100% por intervalo; bins n<100 sombreados.",
                transform=ax.transAxes,
                fontsize=9,
                color="#475569",
            )
            ax.set_xlabel("Intervalo", fontsize=10, labelpad=8, color="#475569")
            ax.set_ylabel("Composición de menciones de skills", fontsize=10, labelpad=8, color="#475569")
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            step = max(1, math.ceil(len(unique_bins) / 12))
            ticks = list(range(0, len(unique_bins), step))
            if ticks[-1] != len(unique_bins) - 1:
                ticks.append(len(unique_bins) - 1)
            ax.set_xticks(ticks)
            ax.set_xticklabels([unique_bins[i] for i in ticks])
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, frameon=False, fontsize=8)
            status = "saved"
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#cbd5e1")
        ax.spines["bottom"].set_color("#cbd5e1")
        ax.grid(axis="y", linestyle="--", alpha=0.5, color="#cbd5e1")
        ax.grid(visible=False, axis="x")
        ax.tick_params(axis="x", rotation=45, colors="#475569", labelsize=8)
        ax.tick_params(axis="y", colors="#475569", labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return status


def _top_skills_overall(domain_skill_monthly: pd.DataFrame, domain: str, n: int = 8) -> str:
    subset = domain_skill_monthly[domain_skill_monthly["domain"].astype(str) == domain]
    if subset.empty:
        return "_No skill data._"
    top = subset.groupby("skill")["skill_job_count"].sum().sort_values(ascending=False).head(n)
    return "; ".join(f"{skill} ({int(count)})" for skill, count in top.items())


def _write_report(
    report_path: Path,
    manifest: dict[str, Any],
    domain_counts: pd.DataFrame,
    domain_skill_monthly: pd.DataFrame,
    plot_records: list[dict[str, Any]],
) -> None:
    lines = [
        "# Skill Evolution (Local TF-IDF, no LLM/Qwen)",
        "",
        "## Command",
        f"`{manifest['command_used']}`",
        "",
        "## Dataset",
        f"- Input: `{manifest['input_path']}`",
        f"- Date range: {manifest['date_range']['min']} to {manifest['date_range']['max']}",
        f"- Rows input: {manifest['input_row_count']}",
        f"- Rows used: {manifest['selected_row_count']}",
        f"- Time bin: `{manifest['bin_size']}`",
        f"- Time column: `{manifest['schema_mapping']['time_column']}`",
        f"- Skill column: `{manifest['schema_mapping'].get('skill_column')}`",
        "",
        "## Domain Assignment",
        (
            "Domains are assigned with TF-IDF + SVD job embeddings matched against domain "
            "description embeddings (cosine similarity), with a keyword fallback. This replaces "
            "the Qwen embedding step from `notebooks/cluster_skill_timelines_colab.ipynb` for "
            "local/offline use. No LLM enrichment is used; skills come only from "
            f"`{manifest['schema_mapping'].get('skill_column')}`."
        ),
        "",
        _format_table(domain_counts, ["domain", "domain_label", "n_jobs", "n_uncertain"], 10),
        "",
        "## Skill Evolution Plots",
    ]
    for record in plot_records:
        if record["png"]:
            lines.append(f"- {record['domain_label']}: `{record['png']}`")
        else:
            lines.append(f"- {record['domain_label']}: insufficient data (`{record['note']}`)")

    lines.extend(["", "## Top Skills per Domain (overall)", "| domain | top_skills |", "| --- | --- |"])
    for domain in TARGET_DOMAINS:
        lines.append(f"| {DOMAIN_LABELS[domain]} | {_top_skills_overall(domain_skill_monthly, domain)} |")

    lines.extend(
        [
            "",
            "## Output Tables",
            f"- Domain assignments: `{manifest['output_paths']['domain_assignments']}`",
            f"- Skill long table: `{manifest['output_paths']['skill_long']}`",
            f"- Domain skill monthly shares: `{manifest['output_paths']['domain_skill_monthly']}`",
            "",
            "## Reliability",
            f"- Label: `{manifest['reliability']['label']}`",
        ]
    )
    warnings = manifest["reliability"].get("warnings", [])
    lines.extend([f"- {warning}" for warning in warnings] if warnings else ["- No reliability warnings."])

    lines.extend(["", "## Limitations"])
    lines.extend([f"- {item}" for item in manifest["limitations"]])
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_skill_evolution(
    input_path: Path,
    outdir: Path,
    bin_size: str = "W",
    top_n_skills: int = 12,
    max_rows: int | None = 100000,
    random_state: int = 42,
    time_column: str | None = None,
    confidence_threshold: float = 0.05,
    margin_threshold: float = 0.01,
    command_used: str | None = None,
) -> SkillEvolutionResult:
    """Run local skill-share evolution analytics and write all report artifacts."""
    started_at = time.perf_counter()
    outdir.mkdir(parents=True, exist_ok=True)
    df_input = pd.read_parquet(input_path)
    if "job_id" not in df_input.columns:
        raise ValueError("Input data must contain a 'job_id' column.")

    schema = detect_temporal_cluster_schema(df_input, time_column=time_column)
    selected = _select_rows(df_input, max_rows=max_rows, random_state=random_state)
    selected["_analysis_text"] = _join_text_columns(selected, schema["text_columns"])
    selected["_posted_at"] = _parse_datetime(selected[schema["time_column"]])
    selected = selected[selected["_posted_at"].notna()].reset_index(drop=True)
    selected["time_bin"] = _time_bins(selected["_posted_at"], bin_size)

    embeddings, vectorizer, svd, _, embedding_meta = build_temporal_cluster_embeddings(
        selected["_analysis_text"],
        embedding="tfidf_svd",
        random_state=random_state,
    )

    job_domains = assign_job_domains(
        selected,
        job_embeddings=embeddings,
        vectorizer=vectorizer,
        svd=svd,
        text_column="_analysis_text",
        skill_column=schema["skill_column"],
        confidence_threshold=confidence_threshold,
        margin_threshold=margin_threshold,
    )
    job_domains.insert(0, "job_id", selected["job_id"].to_numpy())
    job_domains["time_bin"] = selected["time_bin"].to_numpy()
    job_domains = job_domains[DOMAIN_ASSIGNMENT_COLUMNS]

    job_skills_long = build_skill_long_table(selected, schema["skill_column"])
    domain_skill_monthly = compute_domain_skill_monthly(job_skills_long, job_domains)

    domain_counts = (
        job_domains.groupby("domain", observed=False)
        .agg(n_jobs=("job_id", "nunique"), n_uncertain=("is_uncertain", "sum"))
        .reset_index()
    )
    domain_counts["domain_label"] = domain_counts["domain"].map(DOMAIN_LABELS)
    domain_counts["domain"] = domain_counts["domain"].astype(str)
    domain_counts["n_jobs"] = domain_counts["n_jobs"].astype(int)
    domain_counts["n_uncertain"] = domain_counts["n_uncertain"].astype(int)

    domain_assignments_path = outdir / "domain_assignments.parquet"
    skill_long_path = outdir / "skill_long.parquet"
    domain_skill_monthly_path = outdir / "domain_skill_monthly.parquet"
    domain_counts_path = outdir / "domain_counts.parquet"
    manifest_path = outdir / "manifest.json"
    report_path = outdir / "report.md"

    job_domains.to_parquet(domain_assignments_path, index=False)
    job_skills_long.to_parquet(skill_long_path, index=False)
    domain_skill_monthly.to_parquet(domain_skill_monthly_path, index=False)
    domain_counts.to_parquet(domain_counts_path, index=False)

    plot_records: list[dict[str, Any]] = []
    generated_paths = [
        domain_assignments_path,
        skill_long_path,
        domain_skill_monthly_path,
        domain_counts_path,
    ]
    for domain in TARGET_DOMAINS:
        domain_df = domain_skill_monthly[domain_skill_monthly["domain"].astype(str) == domain]
        png_path = outdir / f"skill_evolution_{domain}.png"
        note_path = outdir / f"skill_evolution_{domain}_insufficient_data.txt"
        status = _write_skill_evolution_plot(domain_df, DOMAIN_LABELS[domain], png_path, top_n=top_n_skills)
        if status == "saved":
            generated_paths.append(png_path)
            plot_records.append({"domain": domain, "domain_label": DOMAIN_LABELS[domain], "png": str(png_path), "note": None})
        else:
            note_path.write_text(f"No skill timeline data for {DOMAIN_LABELS[domain]}.\n", encoding="utf-8")
            generated_paths.append(note_path)
            plot_records.append({"domain": domain, "domain_label": DOMAIN_LABELS[domain], "png": None, "note": str(note_path)})

    time_bin_counts = {str(k): int(v) for k, v in selected.groupby("time_bin").size().to_dict().items()}
    reliability = build_reliability_assessment(time_bin_counts)

    valid_dates = selected["_posted_at"].dropna()
    limitations = [
        "TF-IDF + SVD domain assignment is a CPU-safe baseline; it is less accurate than the Qwen-based assignment used in the Colab notebook.",
        "Skills are deterministic, derived only from the skills column; no LLM enrichment is performed.",
        "Domain confidence/margin thresholds were tuned for TF-IDF cosine similarities, which run lower than dense sentence-embedding similarities.",
        "Time bins with few postings can make skill shares noisy.",
    ]
    if embedding_meta.get("embedding_fallback_reason"):
        limitations.append("TF-IDF + SVD embedding fell back to plain TF-IDF; see embedding metadata.")

    manifest: dict[str, Any] = {
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "input_path": str(input_path),
        "input_row_count": int(len(df_input)),
        "selected_row_count": int(len(selected)),
        "date_range": {
            "min": valid_dates.min().isoformat() if not valid_dates.empty else None,
            "max": valid_dates.max().isoformat() if not valid_dates.empty else None,
        },
        "bin_size": bin_size,
        "top_n_skills": int(top_n_skills),
        "random_seed": int(random_state),
        "confidence_threshold": float(confidence_threshold),
        "margin_threshold": float(margin_threshold),
        "command_used": command_used or "jobsrec skill-evolution",
        "git_commit": _git_commit(),
        "package_versions": _package_versions(),
        "schema_mapping": schema,
        "embedding": embedding_meta,
        "domain_assignment": {
            "method": "tfidf_svd_similarity_with_keyword_fallback",
            "target_domains": TARGET_DOMAINS,
            "domain_labels": DOMAIN_LABELS,
        },
        "domain_counts": domain_counts.to_dict(orient="records"),
        "reliability": reliability,
        "limitations": limitations,
        "runtime_seconds": float(time.perf_counter() - started_at),
        "output_paths": {
            "domain_assignments": str(domain_assignments_path),
            "skill_long": str(skill_long_path),
            "domain_skill_monthly": str(domain_skill_monthly_path),
            "domain_counts": str(domain_counts_path),
            "report": str(report_path),
            "manifest": str(manifest_path),
        },
    }
    _write_report(report_path, manifest, domain_counts, domain_skill_monthly, plot_records)
    generated_paths.extend([report_path, manifest_path])
    manifest["generated_files"] = [str(path) for path in generated_paths]
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    return SkillEvolutionResult(
        output_dir=outdir,
        manifest_path=manifest_path,
        report_path=report_path,
        domain_skill_monthly_path=domain_skill_monthly_path,
        generated_files=[str(path) for path in generated_paths],
        manifest=manifest,
    )
