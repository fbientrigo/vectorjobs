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
# PoC sin red ni credenciales (usa data/sample commiteado)
python scripts/download_dataset.py --source-type sample --sample 100

# Mirror en host permitido (GitHub Release / bucket propio)
python scripts/download_dataset.py --source-type url --url <archivo.zip>

# Kaggle oficial (requiere creds + kaggle.com en allowlist)
python scripts/download_dataset.py --source-type kaggle --sample 100
```

El entorno remoto tiene **allowlist de egress** (github sí, kaggle/HF no por
defecto). Alternativas y configuración: ver `scripts/README_dataset.md`.

## Fuentes de precios AWS (jun-2026)

- S3 Standard us-east-1 tiered: $0.023 / $0.022 / $0.021 por GB-mes (nops.io, cloudzero).
- GPU EC2: −45% jun-2025 (cloudoptimo) y +15% ene-2026 (theregister, itpro).
