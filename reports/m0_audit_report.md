# Milestone 0 Audit Report

## Audit Status: PASS (after minimal patches)

---

### Check 1: Run `python -m pytest -q`
**Status**: PASS
**Evidence**:
```text
============================= test session starts =============================
platform win32 -- Python 3.13.5, pytest-9.0.2, pluggy-1.5.0
rootdir: C:\Users\Asus\Documents\code\jobrecon
configfile: pyproject.toml
testpaths: tests
plugins: anyio-4.11.0, cov-7.1.0
collected 53 items

tests\test_data_contract.py ...............                              [ 28%]
tests\test_job_card.py .................                                 [ 60%]
tests\test_tfidf_retrieval.py .....................                      [100%]

============================= 53 passed in 4.38s ==============================
```

### Check 2 & 3: Inspect source and test files
**Status**: PASS
**Evidence**: Reviewed code architecture. Clean encapsulation between data loading (`load.py`), schemas (`schema.py`), text generation (`job_card.py`), vectorization (`tfidf.py`), and retrieval (`retrieval.py`). Tests cover positive and negative pathways accurately without network requests or large assets.

### Check 4: Verify tests are not tautological or too weak
**Status**: PASS
**Evidence**: `tests/test_tfidf_retrieval.py` checks explicit mathematical edge-cases like unit interval scores (`0.0 <= score <= 1.0`), self-match omission assertions, and exception types on input violations. `test_job_card.py` correctly checks section ordering and blank field omission.

### Check 5: Verify core functions return structured objects and do not print
**Status**: PASS
**Evidence**: Core functions return robust dataclasses (`SilverResult`, `ValidationResult`, `TfidfArtifacts`, `RetrievalResult`, `ScoredJob`). Validated via `findstr /s /i /c:"print(" src\jobsrec\*.py` which yielded no hits in core logic. The CLI (`cli.py`) handles output emission cleanly via `click.echo`.

### Check 6: Verify TF-IDF backend rejects a single raw string and accepts `list[str]`
**Status**: PASS
**Evidence**: Built-in helper `_validate_document_list(documents: Any)` inside `tfidf.py` raises `TypeError` explicitly if `isinstance(documents, str)`. Covered reliably by unit test `test_rejects_bare_string`.

### Check 7: Verify retrieval excludes self-matches
**Status**: PASS
**Evidence**: Handled robustly in `retrieval.py` through manual sentinel override: `sims[query_idx] = -1.0`. Covered by `test_results_never_contain_query`.

### Check 8: Verify retrieval does not build a dense NxN similarity matrix
**Status**: PASS
**Evidence**: Verified inside `retrieval.py`. `cosine_similarity(query_vec, self._backend.matrix).ravel()` is used correctly. `query_vec` is vector-shaped `(1, V)` while `matrix` is `(N, V)`. Operation results exactly in sparse computing O(N × V) memory bounds without evaluating the full N×N pairwise map.

### Check 9: Verify build-silver path preserves postings even when a job has no mapped skills
**Status**: PASS
**Evidence**: Enforced in `load.py` line 228. The operation `postings.merge(skills_text_series, on="job_id", how="left")` protects against discarding missing skills, followed by `postings["skills_text"].fillna("")` to safely map `NaN` values.

### Check 10: Verify output manifests are written beside generated artifacts and contain required fields
**Status**: PARTIAL -> PATCHED
**Evidence**: Originally, the field `input_path` was missing from `manifest.json` schema creation in both `load.py` and `tfidf.py`.
**Minimal Patches applied**:
- `src/jobsrec/data/load.py`: Added `input_path` argument to `_make_manifest` signature. Populated it passing `str(input_dir)` inside `build_silver`.
- `src/jobsrec/embeddings/tfidf.py`: Added `input_path` argument to `fit_and_save` parameters and inserted `"input_path": str(input_path)` natively into the resulting manifest dict.
- `src/jobsrec/cli.py`: Passed parameter `input_path=silver_path` properly to `fit_and_save` command sequence.
All manifest checkpoints are now correctly generating `created_at`, `input_path`, `output_path`/`output_dir`, row counts, and config dicts.

### Check 11: Verify `.gitignore` excludes specified artifacts
**Status**: PASS
**Evidence**: Visual review of `.gitignore` confirms `data/raw/`, `data/silver/`, `*.parquet`, `*.npz`, `*.joblib`, `*.index`, `.venv`, `.pytest_cache/`, and `kaggle.json` are appropriately targeted for exclusion.

### Check 12: Verify CLI commands documented in README match implementation
**Status**: PASS
**Evidence**: The `README.md` properly documents `build-silver`, `build-tfidf`, and `recommend`. Their specified flags closely mirror the implemented `@click.option` flags inside `cli.py` (`--input-dir`, `--output-dir`, `--config`, `--silver-path`, `--job-id`, `--top-k`).

### Check 13 & 14: Create a tiny synthetic end-to-end fixture directory and run E2E
**Status**: PASS
**Evidence**: Generated minimal synthetic dataset (`tests/fixtures/raw/postings.csv`, `jobs/job_skills.csv`, `mappings/skills.csv`).
Outputs successfully captured:

#### 1. build-silver output
```text
18:26:08 [INFO] jobsrec.data.load: Loaded postings.csv: 3 rows
18:26:08 [INFO] jobsrec.data.load: Loaded job_skills.csv: 4 rows
18:26:08 [INFO] jobsrec.data.load: Loaded skills.csv: 3 rows
18:26:08 [INFO] jobsrec.data.load: Wrote silver Parquet: tests\fixtures\silver\jobs.parquet (3 rows)
18:26:08 [INFO] jobsrec.data.load: Wrote manifest: tests\fixtures\silver\manifest.json
{
  "output_path": "tests\\fixtures\\silver\\jobs.parquet",
  "input_rows": 3,
  "output_rows": 3
}
```

#### 2. build-tfidf output
```text
18:26:13 [INFO] jobsrec.embeddings.tfidf: Fitted TF-IDF: 3 docs x 21 features
18:26:13 [INFO] jobsrec.embeddings.tfidf: Saved TF-IDF artefacts to tests\fixtures\gold
{
  "vectorizer_path": "tests\\fixtures\\gold\\tfidf_vectorizer.joblib",
  "matrix_path": "tests\\fixtures\\gold\\tfidf_matrix.npz",
  "n_docs": 3,
  "vocab_size": 21
}
```

#### 3. recommend output
```json
{
  "query_job_id": 1,
  "results": [
    {
      "rank": 1,
      "job_id": 2,
      "score": 0.944765
    },
    {
      "rank": 2,
      "job_id": 3,
      "score": 0.374477
    }
  ]
}
```
