import pandas as pd
import pytest

from jobsrec.ingest.populares_contracts import (
    PopularesContractError,
    validate_chunks_frame,
    validate_documents_frame,
    validate_embedding_manifest,
)


def _documents_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "doc_id": ["doc-1", "doc-2"],
            "run_id": ["run-1", "run-1"],
            "source_id": ["uls", "uls"],
            "source_name": ["ULS", "ULS"],
            "source_type": ["university", "university"],
            "url": ["https://example.test/1", "https://example.test/2"],
            "title": ["Ingenieria", "Medicina"],
            "program_name": ["Ingenieria", "Medicina"],
            "degree_name": ["Licenciatura", "Licenciatura"],
            "faculty": ["Facultad 1", "Facultad 2"],
            "section_type": ["perfil_egreso", "campo_laboral"],
            "language": ["es", "es"],
            "text_clean": ["Texto uno", "Texto dos"],
            "content_sha256": ["a" * 64, "b" * 64],
            "source_record_id": ["rec-1", "rec-2"],
            "created_at": pd.to_datetime(["2026-01-01", "2026-01-01"]),
            "fetched_at": pd.to_datetime(["2026-01-01", "2026-01-01"]),
            "normalizer_version": ["v1", "v1"],
        }
    )


def _chunks_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "chunk_id": ["chunk-1", "chunk-2", "chunk-3"],
            "doc_id": ["doc-1", "doc-1", "doc-2"],
            "run_id": ["run-1", "run-1", "run-1"],
            "source_id": ["uls", "uls", "uls"],
            "chunk_index": pd.Series([0, 1, 0], dtype="int32"),
            "section_type": ["perfil_egreso", "perfil_egreso", "campo_laboral"],
            "text": ["Chunk uno", "Chunk dos", "Chunk tres"],
            "char_count": pd.Series([9, 9, 10], dtype="int32"),
            "token_estimate": pd.Series([2, 2, 2], dtype="int32"),
            "content_sha256": ["c" * 64, "d" * 64, "e" * 64],
            "chunking_version": ["v1", "v1", "v1"],
            "metadata_json": ["{}", "{}", "{}"],
        }
    )


def test_valid_documents_frame_passes() -> None:
    assert validate_documents_frame(_documents_frame()) == {"row_count": 2}


def test_valid_chunks_frame_passes() -> None:
    assert validate_chunks_frame(_chunks_frame()) == {"row_count": 3}


def test_missing_document_column_fails_clearly() -> None:
    frame = _documents_frame().drop(columns=["text_clean"])

    with pytest.raises(
        PopularesContractError,
        match=r"Missing required columns in documents\.parquet: \['text_clean'\]",
    ):
        validate_documents_frame(frame)


def test_missing_chunk_column_fails_clearly() -> None:
    frame = _chunks_frame().drop(columns=["chunk_id"])

    with pytest.raises(
        PopularesContractError,
        match=r"Missing required columns in chunks\.parquet: \['chunk_id'\]",
    ):
        validate_chunks_frame(frame)


def test_embedding_manifest_validation_passes() -> None:
    manifest = {
        "model_id": "Qwen/Qwen3-Embedding-0.6B",
        "embedding_dim": 1024,
        "dtype": "float32",
        "input_table": "chunks.parquet",
        "chunking_version": "v1",
        "normalizer_version": "v1",
        "created_at": "2026-01-01T00:00:00Z",
        "n_chunks": 3,
    }

    assert validate_embedding_manifest(manifest)["model_id"] == manifest["model_id"]


def test_embedding_manifest_missing_fields_fails_clearly() -> None:
    with pytest.raises(
        PopularesContractError,
        match=r"embedding_manifest\.json missing fields: \['model_id', 'embedding_dim'\]",
    ):
        validate_embedding_manifest(
            {
                "dtype": "float32",
                "input_table": "chunks.parquet",
                "chunking_version": "v1",
                "normalizer_version": "v1",
                "created_at": "2026-01-01T00:00:00Z",
                "n_chunks": 3,
            }
        )
