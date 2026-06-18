#!/usr/bin/env bash
# Regenera TODAS las figuras de la presentacion dentro de figs/.
#
# Dos tipos de figura:
#   1. Reutilizadas desde reports/ (analitica temporal ya calculada) -> se copian.
#   2. Nuevas (market value + costos AWS) -> las generan los scripts de Python.
#
# Tras correr esto, presentations/estado_actual/ es autocontenida: subela a
# Overleaf y compila limpio (Overleaf usa pdfLaTeX, no necesita Python).
#
# Uso:  bash generate_figures.sh        (desde cualquier sitio)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
FIGS="$HERE/figs"
mkdir -p "$FIGS"

# --- 1. Figuras reutilizadas: destino <- origen en reports/ ------------------ #
# Solo las que referencia main.tex (set canonico y minimo).
copy_fig() {  # $1 = destino en figs/, $2 = origen relativo al repo
  local src="$REPO/$2"
  if [ ! -f "$src" ]; then
    echo "ERROR: falta figura de origen: $2" >&2
    echo "       (corre primero la analitica temporal que la genera)" >&2
    exit 1
  fi
  cp "$src" "$FIGS/$1"
  echo "copiada  figs/$1"
}

copy_fig centroid_drift_by_month.png              reports/temporal_tfidf_10k/figures/centroid_drift_by_month.png
copy_fig job_cluster_map_svd.png                  reports/temporal_tfidf_10k/figures/job_cluster_map_svd.png
copy_fig cluster_semantic_trajectory.png          reports/temporal_clusters/cluster_semantic_trajectory.png
copy_fig cluster_share_timeseries.png             reports/temporal_clusters/cluster_share_timeseries.png
copy_fig salary_coverage_by_month.png             reports/temporal_salary_weighted_tfidf_10k/figures/salary_coverage_by_month.png
copy_fig centroid_drift_salary_weighted_by_month.png reports/temporal_salary_weighted_tfidf_10k/figures/centroid_drift_salary_weighted_by_month.png
copy_fig skill_evolution_tech.png                 reports/skill_evolution/skill_evolution_tech.png
copy_fig skill_evolution_health.png               reports/skill_evolution/skill_evolution_health.png

# --- 2. Figuras nuevas: generadas por los scripts --------------------------- #
echo "generando figuras nuevas (matplotlib)..."
( cd "$REPO" && python3 scripts/aws_cost_projection.py && python3 scripts/market_value.py )

echo "OK -> todas las figuras en $FIGS"
