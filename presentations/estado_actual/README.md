# Presentación: estado actual de vectorjobs

Deck Beamer (español) sobre el estado del proyecto: centroides y su deriva
temporal, "market cap" del mercado laboral (volumen × salario), uso de mejores
modelos, y el frente nuevo de almacenamiento + costos AWS.

## Compilar

```bash
bash build.sh          # tectonic -> latexmk -> pdflatex (lo que esté disponible)
```

Genera `main.pdf` (16 slides). Si faltan las figuras nuevas, `build.sh` las
regenera llamando a los scripts.

## Figuras

- Reutilizadas desde `reports/` (centroides, clusters, skills, salario).
- Generadas por scripts (en `figs/`):
  - `market_value_by_sector.png` ← `scripts/market_value.py`
  - `storage_growth_tb.png`, `aws_cost_projection.png` ← `scripts/aws_cost_projection.py`

## Datos (PoC nube)

`scripts/download_dataset.py` descarga el dataset Kaggle
`arshkon/linkedin-job-postings` y construye un sample pequeño y consistente:

```bash
# Descarga completa + sample de 100 filas (requiere credenciales Kaggle)
python scripts/download_dataset.py --sample 100

# Solo (re)construir el sample desde datos ya descargados (sin red)
python scripts/download_dataset.py --skip-download --sample 100
```

Credenciales en la nube: `KAGGLE_USERNAME` + `KAGGLE_KEY` (o `KAGGLE_API_TOKEN`,
o `~/.kaggle/kaggle.json`).

## Fuentes de precios AWS (jun-2026)

- S3 Standard us-east-1 tiered: $0.023 / $0.022 / $0.021 por GB-mes (nops.io, cloudzero).
- GPU EC2: −45% jun-2025 (cloudoptimo) y +15% ene-2026 (theregister, itpro).
