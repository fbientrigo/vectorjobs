# Conseguir el dataset en la nube — alternativas

`scripts/download_dataset.py` baja el dataset
`arshkon/linkedin-job-postings` y construye un sample pequeño y consistente
(FK). En una sesión remota hay **dos** muros: (1) credenciales de Kaggle y
(2) la **allowlist de egress** de red del entorno.

> Comprobado en este entorno: `github.com` / `codeload.github.com` /
> `*.githubusercontent.com` responden 200; `kaggle.com` y `huggingface.co`
> devuelven `403 Host not in allowlist`. Por eso una sola ruta es frágil y el
> script soporta varias fuentes (`--source-type`).

## Resumen de opciones

| `--source-type` | Necesita credenciales | Host a permitir | Cuándo usarla |
|---|---|---|---|
| `sample` | No | ninguno (sin red) | PoC garantizado en la nube |
| `url` | No | el del mirror (p.ej. GitHub) | mirror propio en host permitido |
| `hf` | No | `huggingface.co`, `cdn-lfs.huggingface.co` | mirror público HF |
| `kaggle` | Sí | `kaggle.com`, `*.kaggleusercontent.com` | fuente oficial |

### 1. `sample` — sin red (recomendado para PoC)
El fixture `data/sample/` (100 jobs, ya commiteado) **es** la fuente. No
descarga nada, siempre funciona:
```bash
python scripts/download_dataset.py --source-type sample --sample 100
```

### 2. `url` — mirror en un host permitido (recomendado para datos completos)
Publica el dataset comprimido una vez y bájalo por HTTPS plano. Como
`github.com` ya está permitido, un **GitHub Release** es la vía más simple:

```bash
# (una vez) publicar el archivo como asset de un Release del repo
gh release create data-v1 linkedin-job-postings.zip

# en la nube
python scripts/download_dataset.py --source-type url \
  --url https://github.com/fbientrigo/vectorjobs/releases/download/data-v1/linkedin-job-postings.zip
```
Soporta `.zip` y `.tar.gz`, reintentos con backoff, y localiza `postings.csv`
aunque venga anidado. También sirve cualquier bucket (S3/GCS/R2) cuyo dominio
añadas a la allowlist; pásalo por `--url` o `DATASET_URL`.

> Límite de asset de Release: 2 GB por archivo (el drop crudo ~500 MB cabe de
> sobra; comprimido aún menos).

### 3. `hf` — mirror público de Hugging Face (sin credenciales)
Existe `xanderios/linkedin-job-postings` en HF. No requiere credenciales, pero
sí que `huggingface.co` esté en la allowlist:
```bash
pip install huggingface_hub
python scripts/download_dataset.py --source-type hf \
  --hf-repo xanderios/linkedin-job-postings
```

### 4. `kaggle` — fuente oficial
Requiere credenciales **y** `kaggle.com` permitido:
```bash
export KAGGLE_USERNAME=<user>; export KAGGLE_KEY=<key>   # o KAGGLE_API_TOKEN
python scripts/download_dataset.py --source-type kaggle --sample 100
```

## Configurar el entorno remoto

- **Allowlist de egress / secrets**: se definen al crear el entorno de Claude
  Code on the web. Añade el host que vayas a usar (p.ej. `kaggle.com` o
  `huggingface.co`) y, para Kaggle, define `KAGGLE_USERNAME` / `KAGGLE_KEY`
  como secretos. Docs: https://code.claude.com/docs/en/claude-code-on-the-web
- `~/.kaggle/kaggle.json` está en `.gitignore`: las credenciales nunca se
  commitean.

## Recomendación

- **PoC inmediato**: `--source-type sample` (cero fricción).
- **Dataset completo reproducible en la nube**: publica un **GitHub Release** y
  usa `--source-type url` — no depende de credenciales de Kaggle ni de añadir
  `kaggle.com`/`huggingface.co` a la allowlist.
