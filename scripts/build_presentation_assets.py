#!/usr/bin/env python3
"""Build presentation figures from a dataset and package an Overleaf folder.

This is the one-shot command for the Beamer deck in
``presentations/0620_current``. It runs the local analytics pipeline against a
raw LinkedIn-style dataset, generates every PNG referenced by ``main.tex``, and
copies the TeX sources plus figures into a self-contained upload directory.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))

import pandas as pd

from jobsrec.trends.temporal import parse_time_column


REQUIRED_FIGURES = {
    "centroid_drift_by_week.png": ("temporal", "centroid_drift_by_week.png"),
    "job_cluster_map_svd.png": ("temporal", "job_cluster_map_svd.png"),
    "salary_coverage_by_week.png": ("temporal", "salary_coverage_by_week.png"),
    "centroid_drift_salary_weighted_by_week.png": (
        "temporal",
        "centroid_drift_salary_weighted_by_week.png",
    ),
    "cluster_semantic_trajectory.png": ("clusters", "cluster_semantic_trajectory.png"),
    "cluster_share_timeseries.png": ("clusters", "cluster_share_timeseries.png"),
    "skill_evolution_tech.png": ("skills", "skill_evolution_tech.png"),
    "skill_evolution_health.png": ("skills", "skill_evolution_health.png"),
}

GENERATED_FIGURES = [
    "market_value_by_sector.png",
    "storage_growth_tb.png",
    "aws_cost_projection.png",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bronze-db",
        type=Path,
        default=Path("data/bronze/jobs.db"),
        help="Path to bronze SQLite jobs.db.",
    )
    parser.add_argument(
        "--silver-dir",
        type=Path,
        default=Path("data/presentation_silver"),
        help="Where jobsrec build-silver writes jobs.parquet.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports/presentation_assets"),
        help="Working directory for regenerated analytics reports.",
    )
    parser.add_argument(
        "--template-dir",
        type=Path,
        default=Path("presentations/0620_current"),
        help="Presentation source template containing main.tex.",
    )
    parser.add_argument(
        "--overleaf-dir",
        type=Path,
        default=Path("dist/estado_actual_overleaf"),
        help="Self-contained folder to upload to Overleaf.",
    )
    parser.add_argument("--sample-size", type=int, default=10000)
    parser.add_argument("--max-rows", type=int, default=100000)
    parser.add_argument("--time-bin", default="W", choices=["H", "D", "W", "M"])
    parser.add_argument("--time-column", default="first_seen_at")
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--skip-silver", action="store_true", help="Reuse --silver-path instead of building silver.")
    parser.add_argument("--silver-path", type=Path, help="Existing jobs.parquet for --skip-silver.")
    parser.add_argument("--clean", action="store_true", help="Delete previous report/output folders before running.")
    parser.add_argument("--strict-figures", action="store_true", help="Fail if a required deck figure is not generated.")
    parser.add_argument("--min-skill-share-pct", type=float, default=5.0, help="Minimum skill share percentage to keep a skill explicit in plotting.")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("+", " ".join(str(part) for part in cmd), flush=True)
    env = os.environ.copy()
    src_path = str(Path("src").resolve())
    env["PYTHONPATH"] = src_path + os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else src_path
    subprocess.run([str(part) for part in cmd], check=True, env=env)


def clean_path(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def build_silver(args: argparse.Namespace) -> Path:
    if args.skip_silver:
        if not args.silver_path:
            raise SystemExit("--skip-silver requires --silver-path")
        if not args.silver_path.exists():
            raise SystemExit(f"Silver file not found: {args.silver_path}")
        return args.silver_path

    run(
        [
            sys.executable,
            "-m",
            "jobsrec.cli",
            "build-silver",
            "--input-db",
            args.bronze_db,
            "--output-dir",
            args.silver_dir,
        ]
    )
    silver_path = args.silver_dir / "jobs.parquet"
    if not silver_path.exists():
        raise SystemExit(f"build-silver did not produce {silver_path}")
    return silver_path


def jobs_per_month_from_silver(silver_path: Path, time_column: str) -> int:
    df = pd.read_parquet(silver_path, columns=[time_column])
    parsed = parse_time_column(df, time_column)
    counts = parsed.dropna().dt.to_period("M").value_counts()
    if counts.empty:
        return max(int(len(df)), 1)
    return max(int(counts.max()), 1)


def run_analytics(args: argparse.Namespace, silver_path: Path) -> dict[str, Path]:
    temporal_dir = args.reports_dir / "temporal_tfidf"
    temporal_figs = temporal_dir / "figures"
    clusters_dir = args.reports_dir / "temporal_clusters"
    skills_dir = args.reports_dir / "skill_evolution"

    run(
        [
            sys.executable,
            "-m",
            "jobsrec.cli",
            "temporal-demo",
            "--input",
            silver_path,
            "--output-dir",
            temporal_dir,
            "--figures-dir",
            temporal_figs,
            "--sample-size",
            args.sample_size,
            "--centroid-weighting",
            "both",
            "--time-column",
            args.time_column,
            "--time-bin",
            args.time_bin,
        ]
    )
    run(
        [
            sys.executable,
            "-m",
            "jobsrec.cli",
            "temporal-clusters",
            "--input",
            silver_path,
            "--outdir",
            clusters_dir,
            "--bin",
            args.time_bin,
            "--k",
            args.k,
            "--max-rows",
            args.max_rows,
            "--random-state",
            args.random_state,
            "--time-column",
            args.time_column,
        ]
    )
    candidates_file = silver_path.parent / "job_extraction_candidates.parquet"
    cmd = [
        sys.executable,
        "-m",
        "jobsrec.cli",
        "skill-evolution",
        "--input",
        silver_path,
        "--outdir",
        skills_dir,
        "--bin",
        args.time_bin,
        "--max-rows",
        args.max_rows,
        "--random-state",
        args.random_state,
        "--time-column",
        args.time_column,
        "--min-skill-share-pct",
        str(args.min_skill_share_pct),
    ]
    if candidates_file.exists():
        cmd.extend(["--candidates-path", candidates_file])
    run(cmd)
    return {"temporal": temporal_figs, "clusters": clusters_dir, "skills": skills_dir}


def copy_template(template_dir: Path, overleaf_dir: Path) -> None:
    overleaf_dir.mkdir(parents=True, exist_ok=True)
    for item in template_dir.iterdir():
        if item.name in {"figs", "main.pdf"}:
            continue
        dest = overleaf_dir / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)


def placeholder_png(path: Path, title: str, detail: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axis("off")
    ax.text(0.5, 0.6, title, ha="center", va="center", fontsize=16, weight="bold")
    ax.text(0.5, 0.42, detail, ha="center", va="center", fontsize=10, wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def package_figures(
    overleaf_dir: Path,
    figure_sources: dict[str, Path],
    generated_dir: Path,
    strict: bool,
) -> list[dict[str, str]]:
    figs_dir = overleaf_dir / "figs"
    if figs_dir.exists():
        shutil.rmtree(figs_dir)
    figs_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, str]] = []
    for dest_name, (source_group, source_name) in REQUIRED_FIGURES.items():
        source = figure_sources[source_group] / source_name
        dest = figs_dir / dest_name
        if source.exists():
            shutil.copy2(source, dest)
            status = "copied"
        else:
            if strict:
                raise SystemExit(f"Required presentation figure was not generated: {source}")
            placeholder_png(
                dest,
                dest_name,
                f"No source figure was generated at {source}. Check report limitations for this dataset.",
            )
            status = "placeholder"
        records.append({"figure": dest_name, "source": str(source), "status": status})

    for name in GENERATED_FIGURES:
        source = generated_dir / name
        dest = figs_dir / name
        if not source.exists():
            raise SystemExit(f"Expected generated figure is missing: {source}")
        shutil.copy2(source, dest)
        records.append({"figure": name, "source": str(source), "status": "copied"})
    return records


def generate_direct_figures(args: argparse.Namespace, silver_path: Path, figs_work_dir: Path) -> None:
    base_jobs = jobs_per_month_from_silver(silver_path, args.time_column)
    clusters_metrics = args.reports_dir / "temporal_clusters" / "cluster_time_metrics.parquet"
    run(
        [
            sys.executable,
            "scripts/aws_cost_projection.py",
            "--out-dir",
            figs_work_dir,
            "--base-jobs-per-month",
            base_jobs,
        ]
    )
    run(
        [
            sys.executable,
            "scripts/market_value.py",
            "--metrics",
            clusters_metrics,
            "--out-dir",
            figs_work_dir,
        ]
    )


def write_manifest(args: argparse.Namespace, silver_path: Path, figures: list[dict[str, str]]) -> None:
    skills_manifest_path = args.reports_dir / "skill_evolution" / "manifest.json"
    skill_source = "unknown"
    candidates_available = False
    skill_dict_version = None
    min_skill_share_pct = args.min_skill_share_pct
    if skills_manifest_path.exists():
        try:
            skills_manifest = json.loads(skills_manifest_path.read_text(encoding="utf-8"))
            skill_source = skills_manifest.get("skill_source", "skills_text")
            candidates_available = skills_manifest.get("candidates_available", False)
            skill_dict_version = skills_manifest.get("skill_dict_version")
            min_skill_share_pct = skills_manifest.get("min_skill_share_pct", args.min_skill_share_pct)
        except Exception:
            pass

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "skip_silver": bool(args.skip_silver),
        "bronze_db": None if args.skip_silver else str(args.bronze_db),
        "silver_path": str(silver_path),
        "reports_dir": str(args.reports_dir),
        "template_dir": str(args.template_dir),
        "overleaf_dir": str(args.overleaf_dir),
        "sample_size": int(args.sample_size),
        "max_rows": int(args.max_rows),
        "time_bin": args.time_bin,
        "time_column": args.time_column,
        "k": int(args.k),
        "random_state": int(args.random_state),
        "min_skill_share_pct": float(min_skill_share_pct),
        "figures": figures,
        "skill_source": skill_source,
        "candidates_available": candidates_available,
        "skill_dict_version": skill_dict_version,
        "caveat": "Regex skills cover named hard skills only and miss many domain skills.",
    }
    (args.overleaf_dir / "asset_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if args.clean:
        clean_path(args.reports_dir)
        if args.overleaf_dir != args.template_dir:
            clean_path(args.overleaf_dir)
        if not args.skip_silver:
            clean_path(args.silver_dir)

    silver_path = build_silver(args)
    figure_sources = run_analytics(args, silver_path)
    generated_dir = args.reports_dir / "presentation_figs"
    generated_dir.mkdir(parents=True, exist_ok=True)
    generate_direct_figures(args, silver_path, generated_dir)
    copy_template(args.template_dir, args.overleaf_dir)
    figures = package_figures(args.overleaf_dir, figure_sources, generated_dir, strict=args.strict_figures)
    write_manifest(args, silver_path, figures)
    print(f"\nOverleaf upload folder ready: {args.overleaf_dir}")
    print(f"Manifest: {args.overleaf_dir / 'asset_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
