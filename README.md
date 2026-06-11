# Job-Skill Recommendation — Milestone 1

> A Colab-first, test-driven pipeline that converts raw LinkedIn job postings
> into a reproducible **silver dataset** and a **TF-IDF baseline** recommender.
> No GPU, no API keys, no LLM inference required for this milestone.

---

## Repository layout

```
jobrecon/
├── configs/
│   ├── local_6gb.yaml       # local CPU config (memory-capped)
│   └── colab_t4.yaml        # Colab T4 GPU config (GPU used in future milestones)
├── data/
│   ├── raw/                 # ← NOT committed (add to .gitignore)
│   ├── silver/              # ← NOT committed
│   └── gold/                # ← NOT committed
├── docs/specs/
│   └── 01_data_contract.md  # column contracts for every CSV / Parquet
├── src/jobsrec/
│   ├── data/
│   │   ├── schema.py        # Pydantic-free column validation helpers
│   │   ├── load.py          # CSV → DataFrame loaders + silver writer
│   │   └── profile.py       # Silver Parquet data profiler (M1)
│   ├── text/
│   │   └── job_card.py      # deterministic job_card_text builder
│   ├── embeddings/
│   │   └── tfidf.py         # TF-IDF vectoriser wrapper
│   ├── recommend/
│   │   └── retrieval.py     # top-k cosine retrieval (sparse, no NxN matrix)
│   └── cli.py               # Click CLI entry-points
├── tests/
│   ├── fixtures/
│   │   └── kaggle_minimal/  # Realistic 10-posting synthetic fixture (M1)
│   ├── test_data_contract.py
│   ├── test_job_card.py
│   ├── test_tfidf_retrieval.py
│   └── test_m1_profile_and_smoke.py  # M1 profiling + fixture smoke tests
├── reports/
│   ├── m0_audit_report.md
│   ├── m1_data_profile.json          # Generated: silver data profile
│   └── m1_real_data_smoke_report.md  # M1 smoke test report
├── pyproject.toml
└── README.md
```

---

## Quick start — local

```bash
# 1. Clone and install (editable)
git clone <repo-url> jobrecon
cd jobrecon
pip install -e ".[dev]"

# 2. Download Kaggle dataset to data/raw/
#    kaggle datasets download -d arshkon/linkedin-job-postings -p data/raw --unzip

# 3. Build silver dataset
python -m jobsrec.cli build-silver \
    --input-dir  data/raw \
    --output-dir data/silver \
    --config     configs/local_6gb.yaml

# 4. Build TF-IDF index
python -m jobsrec.cli build-tfidf \
    --silver-path data/silver/jobs.parquet \
    --output-dir  data/gold \
    --config      configs/local_6gb.yaml

# 5. Recommend similar jobs
python -m jobsrec.cli recommend \
    --job-id  12345 \
    --top-k   5

# 6. Profile the silver dataset
python -m jobsrec.cli profile-silver \
    --silver-path data/silver/jobs.parquet \
    --output     reports/m1_data_profile.json

# 7. Run tests
pytest -q
```

---

## Quick start — Google Colab

```python
# Cell 1 – install
!pip install -q kaggle
!pip install -q -e /content/jobrecon[dev]

# Cell 2 – authenticate Kaggle and download data
from google.colab import files
files.upload()                       # upload kaggle.json
!mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d arshkon/linkedin-job-postings -p data/raw --unzip

# Cell 3 – build silver + TF-IDF + profile
!python -m jobsrec.cli build-silver   --input-dir data/raw --output-dir data/silver --config configs/colab_t4.yaml
!python -m jobsrec.cli build-tfidf    --silver-path data/silver/jobs.parquet --output-dir data/gold --config configs/colab_t4.yaml
!python -m jobsrec.cli profile-silver --silver-path data/silver/jobs.parquet --output reports/m1_data_profile.json

# Cell 4 – recommend
!python -m jobsrec.cli recommend --job-id 12345 --top-k 5
```

---

## Configuration knobs

| Key | local_6gb | colab_t4 |
|-----|-----------|----------|
| `chunksize` | 50 000 | 100 000 |
| `tfidf.max_features` | 30 000 | 60 000 |
| `tfidf.ngram_range` | [1,2] | [1,2] |
| `retrieval.top_k_default` | 10 | 10 |

---

## Data contract

See [`docs/specs/01_data_contract.md`](docs/specs/01_data_contract.md) for the
full column-level contract of every input CSV and the silver Parquet schema.

---

## Testing

```bash
pytest -q                    # fast, no internet, no GPU
pytest --cov=jobsrec -q      # with coverage
```

All tests use **tiny synthetic fixtures** — the Kaggle dataset is never required
to run the test suite.

---

## Milestone 1 — Real-data smoke test & data profiling

Milestone 1 validates the M0 pipeline against realistic synthetic Kaggle-structured
data and adds a data profiling command.

### What's new

- **`src/jobsrec/data/profile.py`** — `profile_silver()` / `profile_silver_from_path()` functions
  that summarise posting counts, skill statistics, datetime parse rates, and column availability.
- **`profile-silver` CLI command** — writes a JSON data profile report.
- **`tests/fixtures/kaggle_minimal/`** — 10-posting synthetic fixture with the full Kaggle
  folder layout (`postings.csv`, `jobs/job_skills.csv`, `mappings/skills.csv`).
- **33 new tests** in `test_m1_profile_and_smoke.py` covering profile correctness, listed_time
  parse rate, and end-to-end fixture smoke tests.

### Milestone 1 CLI workflow

```bash
# 1. Build silver from synthetic fixture
python -m jobsrec.cli build-silver \
    --input-dir tests/fixtures/kaggle_minimal \
    --output-dir scratch/m1_silver

# 2. Build TF-IDF index
python -m jobsrec.cli build-tfidf \
    --silver-path scratch/m1_silver/jobs.parquet \
    --output-dir scratch/m1_gold

# 3. Profile the silver dataset
python -m jobsrec.cli profile-silver \
    --silver-path scratch/m1_silver/jobs.parquet \
    --output reports/m1_data_profile.json

# 4. Get recommendations
python -m jobsrec.cli recommend \
    --job-id 1001 \
    --top-k 3 \
    --gold-dir scratch/m1_gold
```

See [`reports/m1_real_data_smoke_report.md`](reports/m1_real_data_smoke_report.md) for the
full smoke test results and [`reports/m1_data_profile.json`](reports/m1_data_profile.json)
for the generated profile.

---

## Milestone 2 — Dense Embeddings Backend

Milestone 2 adds a dense embedding backend (Qwen3) alongside the existing TF-IDF baseline, preserving the original test-driven architecture.

### What's new

- **`EmbeddingBackend`** interface to support multiple model types.
- **Qwen3 backend** using `sentence-transformers` for dense semantic search.
- **Dense artifact generation** writing embeddings to `.npy`, index to `.parquet`, and metadata to `embedding_manifest.json`.
- **`build-embeddings` and `recommend-dense`** CLI commands.

### Important Notes
- **Testing**: Unit tests use a `FakeDenseEmbeddingBackend` so Qwen model weights are not downloaded during tests.
- **WARNING**: Do not commit generated `.npy` or `.parquet` embedding artifacts to the repository!

### M2 Local CPU Sample Example (1000 rows)

```bash
python -m jobsrec.cli build-embeddings \
    --silver-path data/silver/jobs.parquet \
    --output-dir data/gold/qwen3_0p6b_sample \
    --backend qwen3 \
    --model-name Qwen/Qwen3-Embedding-0.6B \
    --batch-size 8 \
    --device cpu \
    --sample-size 1000

python -m jobsrec.cli recommend-dense \
    --job-id 12345 \
    --top-k 5 \
    --embeddings-dir data/gold/qwen3_0p6b_sample
```

### M2 Colab T4/L4 Example (Full dataset)

```bash
!python -m jobsrec.cli build-embeddings \
    --silver-path data/silver/jobs.parquet \
    --output-dir data/gold/qwen3_0p6b \
    --backend qwen3 \
    --model-name Qwen/Qwen3-Embedding-0.6B \
    --batch-size 32 \
    --device cuda

!python -m jobsrec.cli recommend-dense \
    --job-id 12345 \
    --top-k 5 \
    --embeddings-dir data/gold/qwen3_0p6b
```

### Expected Output
The `build-embeddings` command produces a JSON summary containing `embeddings_path`, `index_path`, `manifest_path`, `n_rows`, and `embedding_dim` (e.g. 1536 for Qwen 0.6B).
The `recommend-dense` command returns JSON recommendations sorted by cosine similarity score.

---

## Milestone 2.3 — Temporal Audit and Drift Demo

M2.3 adds a full temporal coverage audit, reliability gates, and a safe
semantic embedding smoke path. The report language is intentionally limited to
temporal comparison / temporal drift demo unless the dataset has at least six
well-supported months.

Current `data/silver/jobs.parquet` coverage is only two months: 2024-03 has 1
row and 2024-04 has 123,848 rows. Treat every current temporal output as
`limited_temporal_coverage`; do not describe it as stable evolution, market
movement, seasonality, or a durable skill trend.

### Temporal audit

```bash
python -m jobsrec.cli temporal-audit \
    --input data/silver/jobs.parquet \
    --output-dir reports/temporal_audit
```

The audit writes `report.md`, `summary.json`, `monthly_counts.parquet`,
`weekly_counts.parquet`, `daily_counts.parquet`, and
`temporal_column_coverage.parquet`. It flags datasets with fewer than six
months, months below 1,000 rows, and noisy first/last comparisons. The audit
also shows that `original_listed_time` gives a wider but sparse view, while
`expiry` is expiration lifecycle timing rather than posting-demand coverage.

### Existing TF-IDF/SVD baseline

```bash
python -m jobsrec.cli temporal-demo \
    --input data/silver/jobs.parquet \
    --sample-size 10000 \
    --representation tfidf_svd \
    --time-column listed_time \
    --output-dir reports/temporal_tfidf_10k
```

To inspect the wider-but-sparse original listing basis:

```bash
python -m jobsrec.cli temporal-demo \
    --input data/silver/jobs.parquet \
    --sample-size 10000 \
    --representation tfidf_svd \
    --time-column original_listed_time \
    --output-dir reports/temporal_original_time_tfidf_10k
```

### Salary-weighted centroid smoke test

```bash
python -m jobsrec.cli temporal-demo \
    --input data/silver/jobs.parquet \
    --sample-size 10000 \
    --representation tfidf_svd \
    --time-column original_listed_time \
    --centroid-weighting both \
    --output-dir reports/temporal_salary_weighted_tfidf_10k
```

Salary-weighted centroids are written alongside the unweighted baseline. They
describe the salary-disclosed USD subset only; current salary coverage is about
29% of rows.

### Safe semantic smoke test

```bash
python -m jobsrec.cli temporal-demo \
    --input data/silver/jobs.parquet \
    --sample-size 1000 \
    --representation semantic_embeddings \
    --embedding-backend mock \
    --embedding-model deterministic-mock \
    --embedding-batch-size 16 \
    --device cpu \
    --output-dir reports/temporal_semantic_mock_1k
```

The mock backend is deterministic and does not download model weights. It is
intended for tests, schema validation, and end-to-end smoke runs.

### Explicit Qwen3 smoke run

Only run this if `sentence-transformers` is installed and you explicitly want
to load the existing Qwen3 backend. On 4 GB VRAM / 8 GB RAM machines, keep the
sample and batch size small:

```bash
python -m jobsrec.cli temporal-demo \
    --input data/silver/jobs.parquet \
    --sample-size 1000 \
    --representation semantic_embeddings \
    --embedding-backend existing_qwen3 \
    --embedding-model Qwen/Qwen3-Embedding-0.6B \
    --embedding-batch-size 2 \
    --device auto \
    --embedding-cache-dir data/cache/embeddings \
    --output-dir reports/temporal_qwen3_smoke_1k
```

### Legacy explicit output paths

```bash
python -m jobsrec.cli temporal-demo \
    --silver-path data/silver/jobs.parquet \
    --output-dir data/gold/trends_10k \
    --figures-dir reports/figures_10k \
    --report-path reports/m2_2_temporal_demo_10k.md \
    --sample-size 10000 \
    --sampling-mode temporal-stride \
    --representation tfidf_svd \
    --time-column listed_time \
    --centroid-weighting unweighted
```

The default `temporal-stride` sampler sorts valid `listed_time` rows and keeps
coverage across the available month range, including at least one row per
available month when the sample size allows. See
[`docs/specs/03_temporal_trends_prototype.md`](docs/specs/03_temporal_trends_prototype.md)
for the output contract and interpretation rules.

FAISS is not part of M2.3. These commands compute monthly aggregate centroids
and small smoke-test similarities, so an approximate nearest-neighbor index is
not needed yet.

Bootstrap stability for drift and skill growth is deferred in M2.3. Add a
small explicit seed list before using confidence intervals in reports.

Next minimum step: keep `temporal-audit` in front of any temporal demo, preserve
the TF-IDF/SVD baseline, use `original_listed_time` only as a wider-but-sparse
comparison basis, and use deterministic mock semantic runs before any explicit
small Qwen3 run.

---

## Roadmap

| Milestone | Description |
|-----------|-------------|
| **M0** ✅ | TF-IDF baseline |
| **M1** ✅ | Real Kaggle data smoke test + data profiling |
| **M2** ✅ | Dense Embeddings Backend |
| M3 | Open-weight LLM skill extraction (vLLM / Ollama) |
| M4 | Evaluation harness + retrieval metrics |
