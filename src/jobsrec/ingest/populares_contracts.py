"""Contracts for Parquet outputs produced by populares-scraper."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_integer_dtype

DOCUMENT_REQUIRED_COLUMNS = [
    "doc_id",
    "run_id",
    "source_id",
    "source_name",
    "source_type",
    "url",
    "title",
    "program_name",
    "degree_name",
    "faculty",
    "section_type",
    "language",
    "text_clean",
    "content_sha256",
    "source_record_id",
    "created_at",
    "fetched_at",
    "normalizer_version",
]

CHUNK_REQUIRED_COLUMNS = [
    "chunk_id",
    "doc_id",
    "run_id",
    "source_id",
    "chunk_index",
    "section_type",
    "text",
    "char_count",
    "token_estimate",
    "content_sha256",
    "chunking_version",
    "metadata_json",
]

EMBEDDING_MANIFEST_REQUIRED_FIELDS = [
    "model_id",
    "embedding_dim",
    "dtype",
    "input_table",
    "chunking_version",
    "normalizer_version",
    "created_at",
    "n_chunks",
]


class PopularesContractError(ValueError):
    """Raised when populares-scraper output does not match the M0 contract."""


def validate_documents_frame(df: pd.DataFrame) -> dict[str, int]:
    """Validate documents.parquet columns and basic dtypes."""
    _require_columns(df, DOCUMENT_REQUIRED_COLUMNS, "documents.parquet")
    _require_datetime_columns(df, ["created_at", "fetched_at"], "documents.parquet")
    return {"row_count": int(len(df))}


def validate_chunks_frame(df: pd.DataFrame) -> dict[str, int]:
    """Validate chunks.parquet columns and basic dtypes."""
    _require_columns(df, CHUNK_REQUIRED_COLUMNS, "chunks.parquet")
    _require_integer_columns(
        df,
        ["chunk_index", "char_count", "token_estimate"],
        "chunks.parquet",
    )
    return {"row_count": int(len(df))}


def validate_embedding_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate embedding_manifest.json field presence and scalar counts."""
    missing = [
        field for field in EMBEDDING_MANIFEST_REQUIRED_FIELDS if field not in manifest
    ]
    if missing:
        raise PopularesContractError(
            f"embedding_manifest.json missing fields: {missing}"
        )
    for field in ["embedding_dim", "n_chunks"]:
        if not isinstance(manifest[field], int):
            raise PopularesContractError(
                f"embedding_manifest.json field {field!r} must be an integer"
            )
    return dict(manifest)


def _require_columns(
    df: pd.DataFrame,
    required: list[str],
    table_name: str,
) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise PopularesContractError(
            f"Missing required columns in {table_name}: {missing}"
        )


def _require_integer_columns(
    df: pd.DataFrame,
    columns: list[str],
    table_name: str,
) -> None:
    bad = [column for column in columns if not is_integer_dtype(df[column])]
    if bad:
        raise PopularesContractError(
            f"Invalid integer columns in {table_name}: {bad}"
        )


def _require_datetime_columns(
    df: pd.DataFrame,
    columns: list[str],
    table_name: str,
) -> None:
    bad = [column for column in columns if not is_datetime64_any_dtype(df[column])]
    if bad:
        raise PopularesContractError(
            f"Invalid timestamp columns in {table_name}: {bad}"
        )
