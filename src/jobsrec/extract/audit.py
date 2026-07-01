"""M0.7: Baseline quality audit for deterministic extraction."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


AUDIT_VERSION = "baseline_regex_v0_1"


def _skill_counter_and_jobs(candidates: pd.DataFrame) -> tuple[Counter[str], set[str]]:
    skill_counter: Counter[str] = Counter()
    jobs_with_skills: set[str] = set()
    for _, row in candidates.iterrows():
        normed = json.loads(row["skills_normalized"])
        if normed:
            jobs_with_skills.add(row["job_id"])
            skill_counter.update(normed)
    return skill_counter, jobs_with_skills


def compute_audit_metrics(
    silver: pd.DataFrame,
    candidates: pd.DataFrame,
) -> dict[str, Any]:
    total_jobs = len(silver)
    total_candidates = len(candidates)

    per_job = candidates.groupby("job_id").size()
    mean_cands = float(per_job.mean()) if len(per_job) else 0.0
    median_cands = float(per_job.median()) if len(per_job) else 0.0
    jobs_with_candidates = int(per_job.shape[0])

    skill_counter, jobs_with_skills = _skill_counter_and_jobs(candidates)

    # skill hits grouped by section_name
    section_skill_counts: dict[str, int] = {}
    for _, row in candidates.iterrows():
        normed = json.loads(row["skills_normalized"])
        if normed:
            sec = row.get("section_name") or ""
            section_skill_counts[sec] = section_skill_counts.get(sec, 0) + len(normed)

    # industry coverage
    industry_coverage: dict[str, dict[str, Any]] = {}
    if "company_industry" in silver.columns:
        jobs_with_skills_set = jobs_with_skills
        by_industry = silver.groupby("company_industry")["job_id"].apply(set)
        for ind, job_ids in by_industry.items():
            total = len(job_ids)
            with_skills = len(job_ids & jobs_with_skills_set)
            industry_coverage[str(ind)] = {
                "total_jobs": total,
                "jobs_with_skills": with_skills,
                "pct": round(with_skills / total * 100, 1) if total else 0.0,
            }

    source_counts = candidates["candidate_source"].value_counts().to_dict()

    top_50_texts = (
        candidates["candidate_text"]
        .str.lower()
        .str.strip()
        .value_counts()
        .head(50)
        .to_dict()
    )

    parse_errors = (
        int(silver["company_parse_error"].sum())
        if "company_parse_error" in silver.columns
        else 0
    )

    from jobsrec.data.schema import SILVER_SCHEMA_VERSION
    from jobsrec.extract.candidates import EXTRACTION_SCHEMA_VERSION
    from jobsrec.extract.skills import SKILL_DICT_VERSION

    return {
        "audit_version": AUDIT_VERSION,
        "total_jobs": total_jobs,
        "total_candidates": total_candidates,
        "mean_candidates_per_job": round(mean_cands, 2),
        "median_candidates_per_job": round(median_cands, 2),
        "jobs_with_candidates": jobs_with_candidates,
        "jobs_with_regex_skills": len(jobs_with_skills),
        "jobs_with_regex_skills_pct": (
            round(len(jobs_with_skills) / total_jobs * 100, 1) if total_jobs else 0.0
        ),
        "company_parse_errors": parse_errors,
        "silver_schema_version": SILVER_SCHEMA_VERSION,
        "extraction_schema_version": EXTRACTION_SCHEMA_VERSION,
        "skill_dict_version": SKILL_DICT_VERSION,
        "skill_counts": dict(skill_counter.most_common()),
        "skill_counts_by_section": section_skill_counts,
        "candidate_counts_by_source": {str(k): int(v) for k, v in source_counts.items()},
        "jobs_with_skills_by_industry": industry_coverage,
        "top_50_candidate_texts": {str(k): int(v) for k, v in top_50_texts.items()},
    }


def write_json_report(metrics: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def write_markdown_report(metrics: dict[str, Any], output_path: Path) -> None:
    lines: list[str] = [
        f"# Baseline Extraction Quality Audit — {metrics['audit_version']}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total jobs | {metrics['total_jobs']:,} |",
        f"| Total candidate rows | {metrics['total_candidates']:,} |",
        f"| Mean candidates / job | {metrics['mean_candidates_per_job']} |",
        f"| Median candidates / job | {metrics['median_candidates_per_job']} |",
        f"| Jobs with ≥1 candidate | {metrics['jobs_with_candidates']:,} |",
        f"| Jobs with ≥1 regex skill | {metrics['jobs_with_regex_skills']:,} ({metrics['jobs_with_regex_skills_pct']}%) |",
        f"| Company parse errors | {metrics['company_parse_errors']} |",
        f"| Silver schema version | {metrics['silver_schema_version']} |",
        f"| Extraction schema version | {metrics['extraction_schema_version']} |",
        f"| Skill dictionary version | {metrics['skill_dict_version']} |",
        "",
        "## Top Skills",
        "",
        "| Skill | Count |",
        "|-------|-------|",
    ]
    for skill, cnt in list(metrics["skill_counts"].items())[:30]:
        lines.append(f"| {skill} | {cnt:,} |")
    lines += [
        "",
        "## Candidate Sources",
        "",
        "| Source | Count |",
        "|--------|-------|",
    ]
    for src, cnt in sorted(
        metrics["candidate_counts_by_source"].items(), key=lambda x: -x[1]
    ):
        lines.append(f"| {src} | {cnt:,} |")
    lines += [
        "",
        "## Skill Counts by Section",
        "",
        "| Section | Skill Hits |",
        "|---------|------------|",
    ]
    for sec, cnt in sorted(
        metrics["skill_counts_by_section"].items(), key=lambda x: -x[1]
    ):
        lines.append(f"| {sec or '(none)'} | {cnt:,} |")

    if metrics["jobs_with_skills_by_industry"]:
        lines += [
            "",
            "## Industry Coverage",
            "",
            "| Industry | Total Jobs | With Skills | % |",
            "|----------|-----------|-------------|---|",
        ]
        for ind, data in sorted(
            metrics["jobs_with_skills_by_industry"].items(),
            key=lambda x: -x[1]["total_jobs"],
        )[:25]:
            lines.append(
                f"| {ind} | {data['total_jobs']:,} | {data['jobs_with_skills']:,} | {data['pct']}% |"
            )

    lines += [
        "",
        "## Warnings / Caveats",
        "",
        "- `skills_regex_raw` and `skills_normalized` are JSON-encoded `list[str]`; always parse with `json.loads`.",
        f"- {metrics['jobs_with_regex_skills_pct']}% of jobs have at least one regex skill — deterministic baseline before ML.",
        "- Industry labels come from scraped `company_industry` and may be noisy.",
    ]
    if metrics["company_parse_errors"]:
        lines.append(f"- {metrics['company_parse_errors']} jobs had company parse errors.")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_stratified_sample(
    silver: pd.DataFrame,
    candidates: pd.DataFrame,
    sample_size: int = 300,
) -> pd.DataFrame:
    """Return a stratified sample for manual review.

    Strata: has/lacks skills, candidate_source, section_name, industry.
    """
    from jobsrec.extract.candidates import get_stratified_partitions

    parts = get_stratified_partitions(
        silver=silver,
        candidates=candidates,
        sample_size=sample_size,
        sources=("title", "li", "paragraph"),
        sections=("requisitos", "habilidades", "conocimientos", "funciones", "responsabilidades"),
        industries=("Salud", "Retail", "Minería", "Industrial", "Tech", "Telecomunicaciones", "Finanzas", "Banca", "Educación"),
    )

    if not parts:
        return candidates.head(sample_size)

    # ponytail: keep explicit strata first, then fill to requested size from the pool.
    result = pd.concat(parts).drop_duplicates(subset=["job_id", "candidate_index"])
    if len(result) < sample_size:
        result = pd.concat([result, candidates]).drop_duplicates(
            subset=["job_id", "candidate_index"]
        )
    return result.head(sample_size)

