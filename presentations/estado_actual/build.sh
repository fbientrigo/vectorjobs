#!/usr/bin/env bash
# Compila la presentacion Beamer a main.pdf.
# Intenta, en orden: tectonic -> latexmk -> pdflatex.
# Ejecutar desde cualquier sitio: usa la ruta del propio script.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# Regenera las figuras nuevas si falta matplotlib-output (idempotente).
if [ ! -f figs/aws_cost_projection.png ] || [ ! -f figs/market_value_by_sector.png ]; then
  echo "Generando figuras nuevas..."
  (cd ../.. && python3 scripts/aws_cost_projection.py && python3 scripts/market_value.py)
fi

if command -v tectonic >/dev/null 2>&1; then
  echo "Compilando con tectonic..."
  tectonic main.tex
elif command -v latexmk >/dev/null 2>&1; then
  echo "Compilando con latexmk..."
  latexmk -pdf -interaction=nonstopmode main.tex
elif command -v pdflatex >/dev/null 2>&1; then
  echo "Compilando con pdflatex (2 pasadas)..."
  pdflatex -interaction=nonstopmode main.tex
  pdflatex -interaction=nonstopmode main.tex
else
  echo "ERROR: no hay motor LaTeX (tectonic/latexmk/pdflatex)." >&2
  echo "Instala uno, p.ej.: 'apt-get install texlive-latex-recommended texlive-fonts-recommended' o tectonic." >&2
  exit 1
fi

echo "OK -> $HERE/main.pdf"
