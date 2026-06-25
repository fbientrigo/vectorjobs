# Populares Ingest M0

`populares-scraper` owns university website scraping, browser automation, raw HTML,
normalization, semantic documents, chunking, and Parquet export.

`vectorjobs` consumes only the clean outputs:

- `documents.parquet`
- `chunks.parquet`
- `embedding_manifest.json`

This repo intentionally does not scrape websites or run browsers. The M0 adapter
only validates required columns, reads the Parquet files, and prints a
JSON-serializable summary.

Example:

```bash
uv run python -m jobsrec.cli populares-validate \
  --documents path/to/documents.parquet \
  --chunks path/to/chunks.parquet \
  --manifest path/to/embedding_manifest.json
```

M1 can also build a draft gold dataset for `apolo-rag`:

```bash
uv run python -m jobsrec.cli populares-build-gold \
  --documents path/to/documents.parquet \
  --chunks path/to/chunks.parquet \
  --out data/gold/dev
```

This writes `retrieval_corpus.parquet`, `skill_share_by_period.parquet`, and
`dataset_manifest.json`. Skill share is a tiny deterministic lexical lexicon for
now, not robust skill extraction.

Embeddings, RAG, ranking, vector indexes, and model downloads are later
milestones.
