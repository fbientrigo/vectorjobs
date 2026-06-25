"""Load and summarize clean outputs from populares-scraper."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from jobsrec.ingest.populares_contracts import (
    validate_chunks_frame,
    validate_documents_frame,
    validate_embedding_manifest,
)

GOLD_CONTRACT_VERSION = "0.0.1-draft"

RETRIEVAL_CORPUS_COLUMNS = [
    "chunk_id",
    "doc_id",
    "source_id",
    "source_name",
    "source_type",
    "url",
    "title",
    "section_type",
    "text",
    "language",
    "fetched_at",
    "content_sha256",
    "metadata_json",
]

SKILL_SHARE_COLUMNS = [
    "period",
    "skill_name",
    "skill_family",
    "n_mentions",
    "n_documents",
    "share",
    "source_type",
    "source_id",
    "computed_at",
]

SKILL_LEXICON = [
    ("python", "programming", re.compile(r"\bpython\b", re.IGNORECASE)),
    ("sql", "data", re.compile(r"\bsql\b", re.IGNORECASE)),
    (
        "data analysis",
        "data",
        re.compile(
            r"\b(data analysis|analisis de datos|análisis de datos)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "machine learning",
        "ai",
        re.compile(
            r"\b(machine learning|aprendizaje automatico|aprendizaje automático)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "communication",
        "soft",
        re.compile(r"\b(communication|comunicacion|comunicación)\b", re.IGNORECASE),
    ),
    (
        "project management",
        "management",
        re.compile(
            r"\b(project management|gestion de proyectos|gestión de proyectos)\b",
            re.IGNORECASE,
        ),
    ),
]


def load_populares_documents(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    validate_documents_frame(df)
    return df


def load_populares_chunks(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    validate_chunks_frame(df)
    return df


def load_embedding_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("embedding_manifest.json must contain a JSON object")
    return validate_embedding_manifest(data)


def build_populares_gold(
    documents_path: str | Path,
    chunks_path: str | Path,
    out_dir: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    documents = load_populares_documents(documents_path)
    chunks = load_populares_chunks(chunks_path)
    if manifest_path:
        load_embedding_manifest(manifest_path)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat()

    retrieval_corpus = _build_retrieval_corpus(documents, chunks)
    skill_share = _build_skill_share_by_period(documents, created_at)

    retrieval_path = out / "retrieval_corpus.parquet"
    skill_share_path = out / "skill_share_by_period.parquet"
    manifest_path_out = out / "dataset_manifest.json"
    retrieval_corpus.to_parquet(retrieval_path, index=False)
    skill_share.to_parquet(skill_share_path, index=False)

    from jobsrec import __version__

    manifest = {
        "dataset_id": f"populares-gold-{created_at}",
        "gold_contract_version": GOLD_CONTRACT_VERSION,
        "created_at": created_at,
        "producer": "vectorjobs",
        "producer_version": __version__,
        "retrieval_corpus": "retrieval_corpus.parquet",
        "skill_share_by_period": "skill_share_by_period.parquet",
        "n_chunks": int(len(retrieval_corpus)),
        "n_documents": int(documents["doc_id"].nunique()),
        "embedding_model_id": None,
        "embedding_dim": None,
        "index_type": "lexical",
    }
    manifest_path_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def validate_populares_inputs(
    documents_path: str | Path,
    chunks_path: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    documents = load_populares_documents(documents_path)
    chunks = load_populares_chunks(chunks_path)
    manifest = load_embedding_manifest(manifest_path) if manifest_path else None
    return {
        "documents": {"row_count": int(len(documents))},
        "chunks": {"row_count": int(len(chunks))},
        "manifest": manifest,
    }


def summarize_populares_inputs(
    documents_path: str | Path,
    chunks_path: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    documents = load_populares_documents(documents_path)
    chunks = load_populares_chunks(chunks_path)
    manifest = load_embedding_manifest(manifest_path) if manifest_path else None

    doc_ids = set(documents["doc_id"].dropna())
    summary: dict[str, Any] = {
        "n_documents": int(len(documents)),
        "n_chunks": int(len(chunks)),
        "n_sources": int(documents["source_id"].nunique(dropna=True)),
        "source_ids": sorted(
            str(value) for value in documents["source_id"].dropna().unique()
        ),
        "section_type_counts_documents": _counts(documents["section_type"]),
        "section_type_counts_chunks": _counts(chunks["section_type"]),
        "empty_document_text_count": _empty_text_count(documents["text_clean"]),
        "empty_chunk_text_count": _empty_text_count(chunks["text"]),
        "duplicate_document_content_sha256_count": _duplicate_count(
            documents["content_sha256"]
        ),
        "duplicate_chunk_content_sha256_count": _duplicate_count(
            chunks["content_sha256"]
        ),
        "chunks_without_matching_document_count": int(
            (~chunks["doc_id"].isin(doc_ids)).sum()
        ),
        "manifest_model_id": None,
        "manifest_embedding_dim": None,
        "manifest_input_table": None,
        "manifest_n_chunks": None,
        "manifest_chunk_count_matches_actual": None,
    }
    if manifest is not None:
        manifest_n_chunks = manifest["n_chunks"]
        summary.update(
            {
                "manifest_model_id": manifest["model_id"],
                "manifest_embedding_dim": manifest["embedding_dim"],
                "manifest_input_table": manifest["input_table"],
                "manifest_n_chunks": manifest_n_chunks,
                "manifest_chunk_count_matches_actual": (
                    manifest_n_chunks == len(chunks)
                ),
            }
        )
    return summary


def _build_retrieval_corpus(
    documents: pd.DataFrame,
    chunks: pd.DataFrame,
) -> pd.DataFrame:
    missing_doc_ids = sorted(set(chunks["doc_id"]) - set(documents["doc_id"]))
    if missing_doc_ids:
        raise ValueError(
            f"chunks.parquet references missing documents: {missing_doc_ids}"
        )

    doc_columns = [
        "doc_id",
        "source_name",
        "source_type",
        "url",
        "title",
        "language",
        "fetched_at",
    ]
    merged = chunks.merge(
        documents[doc_columns],
        on="doc_id",
        how="left",
        validate="many_to_one",
    )
    corpus = merged[RETRIEVAL_CORPUS_COLUMNS].copy()
    _require_output_columns(corpus, RETRIEVAL_CORPUS_COLUMNS, "retrieval_corpus.parquet")
    return corpus


def _build_skill_share_by_period(
    documents: pd.DataFrame,
    computed_at: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    docs = documents.copy()
    docs["period"] = pd.to_datetime(docs["fetched_at"]).dt.to_period("M").astype(str)
    group_columns = ["period", "source_type", "source_id"]
    computed_at_ts = pd.Timestamp(computed_at)

    for keys, group in docs.groupby(group_columns, dropna=False):
        period, source_type, source_id = keys
        n_documents = int(group["doc_id"].nunique())
        for skill_name, skill_family, pattern in SKILL_LEXICON:
            matches = group["text_clean"].fillna("").map(
                lambda text: len(pattern.findall(str(text)))
            )
            n_mentions = int(matches.sum())
            if n_mentions == 0:
                continue
            n_matching_documents = int((matches > 0).sum())
            rows.append(
                {
                    "period": str(period),
                    "skill_name": skill_name,
                    "skill_family": skill_family,
                    "n_mentions": n_mentions,
                    "n_documents": n_documents,
                    "share": (
                        float(n_matching_documents / n_documents)
                        if n_documents
                        else 0.0
                    ),
                    "source_type": str(source_type),
                    "source_id": str(source_id),
                    "computed_at": computed_at_ts,
                }
            )

    return pd.DataFrame(rows, columns=SKILL_SHARE_COLUMNS)


def _require_output_columns(
    df: pd.DataFrame,
    columns: list[str],
    filename: str,
) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Could not produce required columns in {filename}: {missing}")


def _counts(series: pd.Series) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in series.value_counts(dropna=False).sort_index().items()
    }


def _empty_text_count(series: pd.Series) -> int:
    return int(series.map(lambda value: pd.isna(value) or str(value).strip() == "").sum())


def _duplicate_count(series: pd.Series) -> int:
    return int(series.dropna().duplicated(keep="first").sum())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Validate populares-scraper Parquet outputs."
    )
    parser.add_argument("--documents", required=True, type=Path)
    parser.add_argument("--chunks", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.out:
        result = build_populares_gold(
            args.documents,
            args.chunks,
            args.out,
            args.manifest,
        )
    else:
        result = summarize_populares_inputs(args.documents, args.chunks, args.manifest)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
