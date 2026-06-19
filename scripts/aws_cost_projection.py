#!/usr/bin/env python3
"""Project storage (TB) and AWS cost as the vectorjobs dataset grows.

Two reproducible figures are written to ``presentations/estado_actual/figs/``:

1. ``storage_growth_tb.png`` -- cumulative stored terabytes vs. time for three
   ingestion-growth scenarios (conservative / base / aggressive).
2. ``aws_cost_projection.png`` -- monthly AWS cost (S3 storage + GPU compute for
   embedding generation), with the *real* recent GPU price changes marked.

All prices and assumptions are parameters at the top of the file so the model
can be re-run with updated numbers.

Price sources (retrieved 2026-06):
* S3 Standard, us-east-1 tiered: $0.023/GB-mo (first 50 TB), $0.022 (next
  450 TB), $0.021 (>500 TB).  -- nops.io / cloudzero S3 pricing guides 2026.
* EC2 GPU compute trajectory: ~45% On-Demand cut on Jun-2025 (p5.48xlarge
  $3.8592/hr -> ~$2.16/hr), then a ~15% increase on Jan-2026 H200 capacity.
  -- cloudoptimo (Jun-2025 cut), theregister/itpro (Jan-2026 hike).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

DEFAULT_OUT_DIR = Path("presentations/estado_actual/figs")

# --- Data-growth assumptions ------------------------------------------------ #
BASE_JOBS_PER_MONTH = 123_849          # current LinkedIn drop (Apr-2024)
KB_PER_JOB = {
    "raw_text": 4.27,                  # 516.8 MB / 123849 jobs
    "qwen_embedding": 6.0,             # 1536-dim float32 = 6144 B
    "silver_index_overhead": 2.0,      # parquet silver + tfidf + indices
}
TOTAL_KB_PER_JOB = sum(KB_PER_JOB.values())

SCENARIOS = {                          # month-over-month ingestion growth
    "Conservador (0%/mes)": 0.00,
    "Base (5%/mes)": 0.05,
    "Agresivo (12%/mes)": 0.12,
}
HORIZON_MONTHS = 36

# --- AWS price model -------------------------------------------------------- #
# S3 Standard tiered $/GB-month (us-east-1, 2026).
S3_TIERS = [(50_000, 0.023), (500_000, 0.022), (float("inf"), 0.021)]  # GB, $/GB

# GPU $/hour timeline with the real recent step changes.
GPU_BASELINE = 3.8592                   # p5.48xlarge pre-Jun-2025 On-Demand
GPU_AFTER_CUT = round(GPU_BASELINE * 0.55, 4)     # -45% (Jun-2025) -> ~2.1226
GPU_AFTER_HIKE = round(GPU_AFTER_CUT * 1.15, 4)   # +15% (Jan-2026)
JOBS_PER_GPU_HOUR = 200_000             # Qwen3-0.6B embedding throughput (est.)

# Monthly axis covering the real historical price moves plus a forward window.
PRICE_START = pd.Timestamp("2024-06-01")


def s3_monthly_cost(total_gb: float) -> float:
    """Tiered S3 Standard storage cost for a given stored volume (per month)."""
    cost, lower = 0.0, 0.0
    for upper, price in S3_TIERS:
        if total_gb <= lower:
            break
        billable = min(total_gb, upper) - lower
        cost += billable * price
        lower = upper
    return cost


def gpu_price(month: pd.Timestamp) -> float:
    if month < pd.Timestamp("2025-06-01"):
        return GPU_BASELINE
    if month < pd.Timestamp("2026-01-01"):
        return GPU_AFTER_CUT
    return GPU_AFTER_HIKE


def storage_series(growth: float, base_jobs_per_month: int, horizon_months: int) -> np.ndarray:
    """Cumulative stored TB over the horizon for a MoM ingestion growth rate."""
    monthly_jobs = base_jobs_per_month * (1 + growth) ** np.arange(horizon_months)
    cumulative_jobs = np.cumsum(monthly_jobs)
    return cumulative_jobs * TOTAL_KB_PER_JOB / (1024 ** 3)  # KB -> TB


def plot_storage_growth(out_dir: Path, base_jobs_per_month: int, horizon_months: int) -> Path:
    months = np.arange(horizon_months)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for label, growth in SCENARIOS.items():
        series = storage_series(growth, base_jobs_per_month, horizon_months)
        ax.plot(
            months,
            series,
            marker="o",
            markersize=3,
            label=label,
        )
        ax.annotate(f"{series[-1]:.2f} TB", (months[-1], series[-1]), textcoords="offset points", xytext=(4, 0), fontsize=8)
    ax.set_title("Almacenamiento acumulado bajo supuestos paramétricos")
    ax.set_xlabel("Meses desde hoy")
    ax.set_ylabel("Terabytes acumulados (TB)")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Escenario de ingesta")
    foot = (f"Estimación paramétrica, no uso observado. ~{TOTAL_KB_PER_JOB:.1f} KB/job (raw {KB_PER_JOB['raw_text']} + "
            f"emb {KB_PER_JOB['qwen_embedding']} + overhead "
            f"{KB_PER_JOB['silver_index_overhead']}); base "
            f"{base_jobs_per_month:,} jobs/mes")
    fig.text(0.01, 0.01, foot, fontsize=7, color="gray")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path = out_dir / "storage_growth_tb.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_cost_projection(out_dir: Path, base_jobs_per_month: int, horizon_months: int) -> Path:
    months = pd.date_range(PRICE_START, periods=horizon_months, freq="MS")
    growth = SCENARIOS["Base (5%/mes)"]
    monthly_jobs = base_jobs_per_month * (1 + growth) ** np.arange(horizon_months)
    cumulative_gb = np.cumsum(monthly_jobs) * TOTAL_KB_PER_JOB / (1024 ** 2)  # KB->GB

    s3_cost = np.array([s3_monthly_cost(gb) for gb in cumulative_gb])
    gpu_hr_price = np.array([gpu_price(m) for m in months])
    compute_cost = monthly_jobs / JOBS_PER_GPU_HOUR * gpu_hr_price

    fig, (ax_price, ax_cost) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [1, 2]})

    ax_price.step(months, gpu_hr_price, where="post", color="tab:red")
    ax_price.set_ylabel("GPU $/hora")
    ax_price.set_title("Costo AWS mensual estimado: S3 + cómputo de embeddings")
    ax_price.grid(True, alpha=0.3)
    ax_price.annotate("-45% (jun-2025)", xy=(pd.Timestamp("2025-06-01"), GPU_AFTER_CUT),
                      xytext=(pd.Timestamp("2025-06-01"), GPU_BASELINE * 0.8),
                      fontsize=8, color="tab:red")
    ax_price.annotate("+15% (ene-2026)", xy=(pd.Timestamp("2026-01-01"), GPU_AFTER_HIKE),
                      xytext=(pd.Timestamp("2026-02-01"), GPU_AFTER_HIKE * 1.05),
                      fontsize=8, color="tab:red")

    ax_cost.stackplot(months, s3_cost, compute_cost,
                      labels=["Almacenamiento S3", "Cómputo GPU (embeddings)"],
                      colors=["tab:blue", "tab:orange"], alpha=0.85)
    for change in (pd.Timestamp("2025-06-01"), pd.Timestamp("2026-01-01")):
        ax_cost.axvline(change, color="tab:red", ls="--", lw=1, alpha=0.6)
    ax_cost.set_ylabel("Costo mensual (USD)")
    ax_cost.set_xlabel("Mes")
    ax_cost.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
    ax_cost.grid(True, alpha=0.3)
    ax_cost.legend(loc="upper left")
    foot = (f"Modelo parcial, no TCO productivo: S3 + compute de embeddings. "
            f"S3 tiered $0.023/0.022/0.021 GB-mo; "
            f"{JOBS_PER_GPU_HOUR:,} jobs/GPU-h; escenario base 5%/mes")
    fig.text(0.01, 0.01, foot, fontsize=7, color="gray")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path = out_dir / "aws_cost_projection.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--base-jobs-per-month", type=int, default=BASE_JOBS_PER_MONTH)
    parser.add_argument("--horizon-months", type=int, default=HORIZON_MONTHS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("Wrote", plot_storage_growth(args.out_dir, args.base_jobs_per_month, args.horizon_months))
    print("Wrote", plot_cost_projection(args.out_dir, args.base_jobs_per_month, args.horizon_months))
    final_tb = storage_series(
        SCENARIOS["Base (5%/mes)"],
        args.base_jobs_per_month,
        args.horizon_months,
    )[-1]
    print(f"Base scenario: ~{final_tb:.2f} TB acumulados tras {args.horizon_months} meses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
