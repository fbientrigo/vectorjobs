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

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

OUT_DIR = Path("presentations/estado_actual/figs")
TOTAL_ROWS = 10_000  # rows used in reports/temporal_clusters/report.md

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


def market_value(share: float, salary: float) -> float:
    """Valor de mercado mensual aproximado = postings x salario mediano."""
    return share * TOTAL_ROWS * salary


def plot_market_value() -> Path:
    labels = [s[0] for s in SECTORS]
    first = np.array([market_value(s[2], s[4]) for s in SECTORS]) / 1e6
    last = np.array([market_value(s[3], s[4]) for s in SECTORS]) / 1e6

    order = np.argsort(last)[::-1]
    labels = [labels[i] for i in order]
    first, last = first[order], last[order]

    y = np.arange(len(labels))
    height = 0.38
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(y + height / 2, first, height, label="Inicio de ventana", color="tab:gray", alpha=0.7)
    ax.barh(y - height / 2, last, height, label="Fin de ventana", color="tab:blue")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Valor de mercado ≈ postings × salario mediano (M USD)")
    ax.set_title("'Market cap' del mercado laboral por sector\n(volumen × salario, ilustrativo)")
    ax.xaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.1f}M"))
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend()
    foot = ("Volumen/share: reports/temporal_clusters (10k filas, abr-2024). "
            "Salario: supuesto por sector. Caveat: limited_temporal_coverage — "
            "recalcular con jobs.parquet completo.")
    fig.text(0.01, 0.01, foot, fontsize=7, color="gray")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path = OUT_DIR / "market_value_by_sector.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Wrote", plot_market_value())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
