#!/usr/bin/env python3
"""'Market cap' del mercado laboral = volumen de postings x salario por sector.

Combina los dos ejes que el proyecto ya mide -- el *volumen* de ofertas por
cluster y el *salario* -- en un único proxy de "valor de mercado" por sector y
su cambio a lo largo de la ventana observada.

Fuente de datos
---------------
* Volumen y share por cluster: tablas reales de
  ``reports/temporal_clusters/report.md`` (12 clusters fijos, 10.000 filas,
  shares de inicio y fin de la ventana abr-2024).
* Salario mediano por sector: vector documentado abajo (USD anual). El dataset
  solo trae salario en el 29% de las filas, así que se usa una referencia por
  sector, claramente etiquetada como supuesto. Recalcular con datos completos
  cuando ``data/silver/jobs.parquet`` esté disponible (salario mediano real por
  cluster).

La figura se marca como *ilustrativa* y arrastra el caveat de cobertura
``limited_temporal_coverage`` (la ventana real es corta).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

DEFAULT_OUT_DIR = Path("presentations/estado_actual/figs")
TOTAL_ROWS = 10_000  # rows used in reports/temporal_clusters/report.md
DEFAULT_MEDIAN_SALARY = 75_000

# (sector, n_jobs, first_share, last_share, salario_mediano_USD)
# n_jobs / shares: reales del report. Salario: supuesto por sector (referencia).
SECTORS = [
    ("Data / Marketing / Negocio", 1179, 0.1099, 0.3704, 85_000),
    ("Gestión / Construcción",     1157, 0.1466, 0.1481, 75_000),
    ("Salud / Enfermería",         1036, 0.1152, 0.0741, 80_000),
    ("Ventas / Clientes",           818, 0.0942, 0.0741, 60_000),
    ("Software / Ingeniería",       787, 0.0733, 0.0741, 108_000),
    ("Retail / Tienda",             631, 0.0524, 0.0583, 38_000),
    ("Mantenimiento / Eléctrico",   578, 0.0707, 0.0370, 55_000),
    ("Contabilidad / Finanzas",     399, 0.0445, 0.0418, 90_000),
    ("Hostelería / Alimentos",      371, 0.0262, 0.0389, 35_000),
]


def market_value(share: float, salary: float, total_rows: int = TOTAL_ROWS) -> float:
    """Valor de mercado mensual aproximado = postings x salario mediano."""
    return share * total_rows * salary


def _load_cluster_metrics(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _sector_rows_from_metrics(
    path: Path,
    min_bin_jobs: int = 100,
) -> tuple[list[tuple[str, int, float, float, float]], str]:
    metrics = _load_cluster_metrics(path)
    required = {"time_bin", "cluster_label", "n_jobs", "share_jobs"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    if metrics.empty:
        raise ValueError(f"{path} has no rows")

    metrics = metrics.copy()
    metrics["time_bin"] = metrics["time_bin"].astype(str)
    totals_by_bin = metrics.groupby("time_bin")["n_jobs"].sum().sort_index()
    eligible_bins = totals_by_bin[totals_by_bin >= min_bin_jobs]
    if eligible_bins.empty:
        eligible_bins = totals_by_bin
    first_bin = str(eligible_bins.index[0])
    last_bin = str(eligible_bins.index[-1])
    total_rows = int(totals_by_bin.max())
    total_rows = max(total_rows, 1)

    if "salary_median" in metrics.columns:
        salary = pd.to_numeric(metrics["salary_median"], errors="coerce")
    else:
        salary = pd.Series([np.nan] * len(metrics), index=metrics.index)
    fallback_salary = float(salary.dropna().median()) if salary.notna().any() else DEFAULT_MEDIAN_SALARY
    metrics["salary_for_value"] = salary.fillna(fallback_salary)

    first = metrics[metrics["time_bin"] == first_bin].set_index("cluster_label")
    last = metrics[metrics["time_bin"] == last_bin].set_index("cluster_label")
    labels = sorted(set(first.index) | set(last.index))

    rows: list[tuple[str, int, float, float, float]] = []
    for label in labels:
        first_row = first.loc[label] if label in first.index else None
        last_row = last.loc[label] if label in last.index else None
        first_share = float(first_row["share_jobs"]) if first_row is not None else 0.0
        last_share = float(last_row["share_jobs"]) if last_row is not None else 0.0
        n_jobs = int(last_row["n_jobs"]) if last_row is not None else int(first_row["n_jobs"])
        salary_value = (
            float(last_row["salary_for_value"])
            if last_row is not None
            else float(first_row["salary_for_value"])
        )
        rows.append((str(label), n_jobs, first_share, last_share, salary_value))

    source_note = (
        f"Volumen/share: {path} ({first_bin} -> {last_bin}). "
        f"Salario: salary_median por cluster; faltantes rellenados con ${fallback_salary:,.0f}; "
        f"bins con <{min_bin_jobs} postings omitidos en endpoints."
    )
    return rows, source_note


def plot_market_value(
    out_dir: Path,
    sectors: list[tuple[str, int, float, float, float]],
    source_note: str,
    total_rows: int = TOTAL_ROWS,
) -> Path:
    labels = [s[0] for s in SECTORS]
    if sectors:
        labels = [s[0] for s in sectors]
    else:
        sectors = SECTORS
    first = np.array([market_value(s[2], s[4], total_rows) for s in sectors]) / 1e6
    last = np.array([market_value(s[3], s[4], total_rows) for s in sectors]) / 1e6

    order = np.argsort(last)[::-1]
    labels = [labels[i] for i in order]
    first, last = first[order], last[order]

    y = np.arange(len(labels))
    height = 0.38
    fig, ax = plt.subplots(figsize=(10.5, 6.3))
    ax.barh(y + height / 2, first, height, label="Inicio de ventana", color="tab:gray", alpha=0.7)
    ax.barh(y - height / 2, last, height, label="Fin de ventana", color="tab:blue")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Proxy de valor = posting share × salario mediano (M USD)")
    ax.set_title("Proxy de valor laboral por sector\n(posting share × salario mediano)")
    ax.xaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.1f}M"))
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend()
    foot = textwrap.fill(source_note, width=130)
    fig.text(0.01, 0.015, foot, fontsize=7, color="gray", va="bottom")
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    path = out_dir / "market_value_by_sector.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, help="cluster_time_metrics.parquet from temporal-clusters.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-bin-jobs", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.metrics:
        sectors, note = _sector_rows_from_metrics(args.metrics, min_bin_jobs=args.min_bin_jobs)
        total_rows = max(sum(s[1] for s in sectors), 1)
    else:
        sectors = SECTORS
        total_rows = TOTAL_ROWS
        note = ("Volumen/share: fallback static demo values. "
                "Run with --metrics reports/.../cluster_time_metrics.parquet for data-specific output.")
    print("Wrote", plot_market_value(args.out_dir, sectors, note, total_rows=total_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
