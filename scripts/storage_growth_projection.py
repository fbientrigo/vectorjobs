#!/usr/bin/env python3
"""Measured storage growth projection for the current vectorjobs artifacts."""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd

DEFAULT_DB = Path("data/bronze/jobs.db")
DEFAULT_SILVER = Path("data/silver/jobs.parquet")
DEFAULT_CANDIDATES = Path("data/silver/job_extraction_candidates.parquet")
DEFAULT_GOLD_DIR = Path("reports/presentation_assets")
DEFAULT_OUT_DIR = Path("reports/storage_projection")
SCENARIOS = {"0%/month": 0.00, "5%/month": 0.05, "12%/month": 0.12}
TOTAL_SCENARIO = "12%/month"
TOTAL_SCENARIO_NOTE = (
    "Worst-case path: each projected month adds 12% more new data than the previous month."
)
SCENARIO_TOTAL_TARGET_GB = 50.0
SCENARIO_TOTAL_MAX_MONTHS = 120
PLOT_FILES = [
    "storage_growth_mb.png",
    "storage_growth_gb.png",
    "storage_total_scenarios_mb.png",
    "storage_total_scenarios_gb.png",
]


@dataclass(frozen=True)
class BronzeMetrics:
    db_size_bytes: int
    jobs_count: int
    observations_count: int
    crawl_runs_count: int
    first_seen_monthly: dict[str, int]


@dataclass(frozen=True)
class StorageAnchors:
    bronze_bytes_per_job: float
    silver_bytes_per_job: float
    fixed_gold_bytes: float
    variable_gold_bytes_per_job: float
    measured: dict[str, Any]


def read_bronze_metrics(db_path: Path) -> BronzeMetrics:
    with sqlite3.connect(db_path) as con:
        tables = {row[0] for row in con.execute("select name from sqlite_master where type='table'")}
        missing = {"jobs", "job_observations", "crawl_runs"} - tables
        if missing:
            raise ValueError(f"Bronze DB is missing required tables: {sorted(missing)}")
        first_seen = con.execute(
            """
            select strftime('%Y-%m', first_seen_at) as month, count(*)
            from (
                select job_id, min(seen_at) as first_seen_at
                from job_observations
                group by job_id
            )
            group by month
            order by month
            """
        ).fetchall()
        return BronzeMetrics(
            db_size_bytes=db_path.stat().st_size,
            jobs_count=int(con.execute("select count(*) from jobs").fetchone()[0]),
            observations_count=int(con.execute("select count(*) from job_observations").fetchone()[0]),
            crawl_runs_count=int(con.execute("select count(*) from crawl_runs").fetchone()[0]),
            first_seen_monthly={str(month): int(count) for month, count in first_seen if month},
        )


def future_base_monthly_jobs(monthly_counts: dict[str, int]) -> int:
    counts = [count for _, count in sorted(monthly_counts.items())]
    if not counts:
        return 1
    tail = counts[1:] if len(counts) > 1 else counts
    return max(round(sum(tail) / len(tail)), 1)


def artifact_bytes(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        if path.is_file():
            total += path.stat().st_size
        elif path.is_dir():
            total += sum(child.stat().st_size for child in path.rglob("*") if child.is_file())
    return total


def silver_rows(silver_path: Path, fallback: int) -> int:
    manifest = silver_path.parent / "manifest.json"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            return max(int(data.get("output_rows") or data.get("input_rows") or fallback), 1)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    try:
        return max(int(len(pd.read_parquet(silver_path, columns=["job_id"]))), 1)
    except Exception:
        return max(int(fallback), 1)


def measure_anchors(
    *,
    bronze: BronzeMetrics,
    silver_path: Path,
    candidates_path: Path,
    gold_dir: Path,
    fixed_gold_bytes: int = 2_000_000,
) -> StorageAnchors:
    rows = silver_rows(silver_path, bronze.jobs_count)
    silver_bytes = artifact_bytes([silver_path, candidates_path])
    gold_bytes = artifact_bytes([gold_dir])
    variable_gold = max((gold_bytes - fixed_gold_bytes) / max(rows, 1), 0.0)
    return StorageAnchors(
        bronze_bytes_per_job=bronze.db_size_bytes / max(bronze.jobs_count, 1),
        silver_bytes_per_job=silver_bytes / max(rows, 1),
        fixed_gold_bytes=float(fixed_gold_bytes),
        variable_gold_bytes_per_job=variable_gold,
        measured={
            "bronze_db_bytes": bronze.db_size_bytes,
            "bronze_jobs": bronze.jobs_count,
            "bronze_observations": bronze.observations_count,
            "bronze_crawl_runs": bronze.crawl_runs_count,
            "first_seen_monthly_jobs": bronze.first_seen_monthly,
            "future_base_monthly_jobs": future_base_monthly_jobs(bronze.first_seen_monthly),
            "silver_rows": rows,
            "silver_artifact_bytes": silver_bytes,
            "gold_bundle_bytes": gold_bytes,
        },
    )


def project_storage(
    *,
    bronze: BronzeMetrics,
    anchors: StorageAnchors,
    horizon_months: int = 36,
    scenarios: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    scenarios = scenarios or {TOTAL_SCENARIO: SCENARIOS[TOTAL_SCENARIO]}
    base_jobs = future_base_monthly_jobs(bronze.first_seen_monthly)
    last_month = max(bronze.first_seen_monthly) if bronze.first_seen_monthly else datetime.now().strftime("%Y-%m")
    start = (pd.Period(last_month, freq="M") + 1).to_timestamp()
    initial_total = (anchors.bronze_bytes_per_job + anchors.silver_bytes_per_job) * bronze.jobs_count
    rows: list[dict[str, Any]] = []

    for scenario, growth in scenarios.items():
        cumulative_jobs = float(bronze.jobs_count)
        gold_total = 0.0
        previous = {"bronze": anchors.bronze_bytes_per_job * cumulative_jobs, "silver": anchors.silver_bytes_per_job * cumulative_jobs, "gold": 0.0, "total": initial_total}
        for index in range(1, horizon_months + 1):
            month_jobs = float(round(base_jobs * (1 + growth) ** (index - 1)))
            cumulative_jobs += month_jobs
            bundles = 1 + (1 if index % 6 == 0 else 0) + (1 if index % 12 == 0 else 0)
            gold_total += bundles * (anchors.fixed_gold_bytes + anchors.variable_gold_bytes_per_job * cumulative_jobs)
            current = {
                "bronze": anchors.bronze_bytes_per_job * cumulative_jobs,
                "silver": anchors.silver_bytes_per_job * cumulative_jobs,
                "gold": gold_total,
            }
            current["total"] = current["bronze"] + current["silver"] + current["gold"]
            row: dict[str, Any] = {
                "scenario": scenario,
                "monthly_growth": growth,
                "month_index": index,
                "month": (start + pd.DateOffset(months=index - 1)).strftime("%Y-%m"),
                "new_jobs": int(month_jobs),
                "cumulative_jobs": int(round(cumulative_jobs)),
                "gold_bundles_added": bundles,
            }
            for layer in ["bronze", "silver", "gold", "total"]:
                row[f"{layer}_bytes"] = current[layer]
                row[f"{layer}_added_bytes"] = current[layer] - previous[layer]
                row[f"{layer}_mb"] = current[layer] / 1_000_000
                row[f"{layer}_added_mb"] = (current[layer] - previous[layer]) / 1_000_000
                row[f"{layer}_gb"] = current[layer] / 1_000_000_000
                row[f"{layer}_added_gb"] = (current[layer] - previous[layer]) / 1_000_000_000
            rows.append(row)
            previous = current
    return rows


def plot_projection(rows: list[dict[str, Any]], path: Path, *, unit: str) -> None:
    suffix = unit.lower()
    layers = ["bronze", "silver", "gold", "total"]
    colors = {"bronze": "tab:brown", "silver": "tab:blue", "gold": "tab:orange", "total": "black"}
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.stackplot(
        df["month_index"],
        *(df[f"{layer}_{suffix}"] for layer in layers[:-1]),
        labels=["Bronze", "Silver", "Gold"],
        colors=[colors[layer] for layer in layers[:-1]],
        alpha=0.78,
    )
    ax.plot(df["month_index"], df[f"total_{suffix}"], color=colors["total"], linewidth=2.5, label="Total")
    ax.set_title(f"Storage projection by layer ({TOTAL_SCENARIO})")
    ax.set_ylabel(f"Cumulative {unit}")
    ax.set_xlabel("Months from projection start")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
    ax.legend(ncol=4, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def scenario_total_horizon(
    bronze: BronzeMetrics,
    anchors: StorageAnchors,
    *,
    target_gb: float = SCENARIO_TOTAL_TARGET_GB,
    max_months: int = SCENARIO_TOTAL_MAX_MONTHS,
) -> dict[str, Any]:
    rows = project_storage(
        bronze=bronze,
        anchors=anchors,
        horizon_months=max_months,
        scenarios={TOTAL_SCENARIO: SCENARIOS[TOTAL_SCENARIO]},
    )
    final = rows[-1]
    for row in rows:
        if row["total_gb"] >= target_gb:
            final = row
            break
    return {
        "target_gb": target_gb,
        "months": int(final["month_index"]),
        "ending_month": final["month"],
        "final_worst_case_gb": final["total_gb"],
    }


def plot_total_scenarios(rows: list[dict[str, Any]], path: Path, *, unit: str) -> None:
    suffix = unit.lower()
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for scenario, group in df.groupby("scenario", sort=False):
        ax.plot(group["month_index"], group[f"total_{suffix}"], linewidth=2.2, label=scenario)
    
    # Mark thresholds (10GB, 20GB, 40GB, 80GB) for the 12%/month scenario
    if TOTAL_SCENARIO in df["scenario"].values:
        df_12 = df[df["scenario"] == TOTAL_SCENARIO]
        x = df_12["month_index"].values
        y = df_12[f"total_{suffix}"].values
        thresholds_gb = [10.0, 20.0, 40.0, 80.0]
        max_y = y.max() if len(y) > 0 else 1.0
        for t_gb in thresholds_gb:
            t_val = t_gb if unit == "GB" else t_gb * 1000.0
            if len(y) > 1 and y[0] <= t_val <= y[-1]:
                x_t = None
                for i in range(len(y) - 1):
                    if y[i] <= t_val <= y[i+1]:
                        t_frac = (t_val - y[i]) / (y[i+1] - y[i])
                        x_t = x[i] + t_frac * (x[i+1] - x[i])
                        break
                if x_t is not None:
                    ax.hlines(
                        y=t_val,
                        xmin=x[0],
                        xmax=x_t,
                        colors="tab:red",
                        linestyles="--",
                        linewidth=1.2,
                        alpha=0.6,
                    )
                    ax.vlines(
                        x=x_t,
                        ymin=0,
                        ymax=t_val,
                        colors="tab:red",
                        linestyles="--",
                        linewidth=1.2,
                        alpha=0.6,
                    )
                    ax.plot(
                        x_t,
                        t_val,
                        marker="o",
                        color="tab:red",
                        markersize=6,
                        zorder=5,
                    )
                    ax.annotate(
                        f"{int(t_gb)} GB @ {x_t:.1f} mo",
                        xy=(x_t, t_val),
                        xytext=(x_t - 0.8, t_val + max_y * 0.015),
                        fontsize=9,
                        fontweight="bold",
                        color="tab:red",
                        ha="right",
                        va="bottom",
                        bbox=dict(
                            boxstyle="round,pad=0.2",
                            fc="white",
                            ec="tab:red",
                            lw=0.8,
                            alpha=0.85,
                        ),
                        zorder=6,
                    )

    ax.set_title("Total storage projection by ingestion-growth scenario")
    ax.set_ylabel(f"Cumulative {unit}")
    ax.set_xlabel("Months from projection start")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_outputs(
    rows: list[dict[str, Any]],
    anchors: StorageAnchors,
    out_dir: Path,
    *,
    scenario_rows: list[dict[str, Any]] | None = None,
    scenario_horizon: dict[str, Any] | None = None,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "monthly_storage_projection.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    for stale in ["storage_growth_mb_log.png", "storage_growth_gb_log.png"]:
        (out_dir / stale).unlink(missing_ok=True)
    for unit, name in [("MB", "storage_growth_mb.png"), ("GB", "storage_growth_gb.png")]:
        plot_projection(rows, out_dir / name, unit=unit)
    scenario_rows = scenario_rows or rows
    for unit, name in [("MB", "storage_total_scenarios_mb.png"), ("GB", "storage_total_scenarios_gb.png")]:
        plot_total_scenarios(scenario_rows, out_dir / name, unit=unit)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "units": {"MB": "1,000,000 bytes", "GB": "1,000,000,000 bytes"},
        "scenarios": SCENARIOS,
        "total_scenario": {
            "label": TOTAL_SCENARIO,
            "monthly_growth": SCENARIOS[TOTAL_SCENARIO],
            "meaning": TOTAL_SCENARIO_NOTE,
        },
        "scenario_total_horizon": scenario_horizon
        or {
            "target_gb": SCENARIO_TOTAL_TARGET_GB,
            "months": max(int(row["month_index"]) for row in scenario_rows),
            "ending_month": scenario_rows[-1]["month"],
            "final_worst_case_gb": scenario_rows[-1]["total_gb"],
        },
        "anchors": {
            **anchors.measured,
            "bronze_bytes_per_job": anchors.bronze_bytes_per_job,
            "silver_bytes_per_job": anchors.silver_bytes_per_job,
            "fixed_gold_bytes": anchors.fixed_gold_bytes,
            "variable_gold_bytes_per_job": anchors.variable_gold_bytes_per_job,
        },
        "outputs": {"csv": str(csv_path), "plots": [str(out_dir / name) for name in PLOT_FILES]},
    }
    manifest_path = out_dir / "storage_projection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"csv": str(csv_path), "manifest": str(manifest_path), **{name: str(out_dir / name) for name in PLOT_FILES}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bronze-db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--silver-path", type=Path, default=DEFAULT_SILVER)
    parser.add_argument("--candidates-path", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--horizon-months", type=int, default=36)
    parser.add_argument("--fixed-gold-bytes", type=int, default=2_000_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bronze = read_bronze_metrics(args.bronze_db)
    anchors = measure_anchors(
        bronze=bronze,
        silver_path=args.silver_path,
        candidates_path=args.candidates_path,
        gold_dir=args.gold_dir,
        fixed_gold_bytes=args.fixed_gold_bytes,
    )
    rows = project_storage(bronze=bronze, anchors=anchors, horizon_months=args.horizon_months)
    scenario_horizon = scenario_total_horizon(bronze, anchors)
    
    # Calculate the horizon for 80 GB so the plots can cover all thresholds up to 80 GB
    scenario_horizon_80 = scenario_total_horizon(bronze, anchors, target_gb=80.0)
    scenario_rows = project_storage(
        bronze=bronze,
        anchors=anchors,
        horizon_months=scenario_horizon_80["months"],
        scenarios=SCENARIOS,
    )
    outputs = write_outputs(rows, anchors, args.out_dir, scenario_rows=scenario_rows, scenario_horizon=scenario_horizon)
    print(f"Wrote {outputs['csv']}")
    print(f"Wrote {outputs['manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
