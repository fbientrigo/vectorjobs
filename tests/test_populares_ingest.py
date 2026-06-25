import json
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from jobsrec.cli import main
from jobsrec.ingest.populares import (
    build_populares_gold,
    load_populares_chunks,
    load_populares_documents,
    summarize_populares_inputs,
)


def _write_populares_files(
    tmp_path: Path,
    *,
    duplicate_hash: bool = False,
    empty_text: bool = False,
    unknown_doc_id: bool = False,
    manifest_n_chunks: int = 3,
) -> tuple[Path, Path, Path]:
    documents_path = tmp_path / "documents.parquet"
    chunks_path = tmp_path / "chunks.parquet"
    manifest_path = tmp_path / "embedding_manifest.json"

    documents = pd.DataFrame(
        {
            "doc_id": ["doc-1", "doc-2"],
            "run_id": ["run-1", "run-1"],
            "source_id": ["uls", "uls"],
            "source_name": ["Universidad de La Serena", "Universidad de La Serena"],
            "source_type": ["university", "university"],
            "url": ["https://example.test/1", "https://example.test/2"],
            "title": ["Perfil de egreso", "Campo laboral"],
            "program_name": ["Ingenieria", "Ingenieria"],
            "degree_name": ["Licenciatura", "Licenciatura"],
            "faculty": ["Ingenieria", "Ingenieria"],
            "section_type": ["perfil_egreso", "campo_laboral"],
            "language": ["es", "es"],
            "text_clean": [
                "Python SQL comunicacion",
                "" if empty_text else "Analisis de datos y gestion de proyectos",
            ],
            "content_sha256": ["a" * 64, "a" * 64 if duplicate_hash else "b" * 64],
            "source_record_id": ["rec-1", "rec-2"],
            "created_at": pd.to_datetime(["2026-01-01", "2026-01-01"]),
            "fetched_at": pd.to_datetime(["2026-01-01", "2026-01-01"]),
            "normalizer_version": ["v1", "v1"],
        }
    )
    chunks = pd.DataFrame(
        {
            "chunk_id": ["chunk-1", "chunk-2", "chunk-3"],
            "doc_id": ["doc-1", "doc-1", "missing-doc" if unknown_doc_id else "doc-2"],
            "run_id": ["run-1", "run-1", "run-1"],
            "source_id": ["uls", "uls", "uls"],
            "chunk_index": pd.Series([0, 1, 0], dtype="int32"),
            "section_type": ["perfil_egreso", "perfil_egreso", "campo_laboral"],
            "text": ["Chunk uno", "Chunk dos", "" if empty_text else "Chunk tres"],
            "char_count": pd.Series([9, 9, 10], dtype="int32"),
            "token_estimate": pd.Series([2, 2, 2], dtype="int32"),
            "content_sha256": [
                "c" * 64,
                "c" * 64 if duplicate_hash else "d" * 64,
                "e" * 64,
            ],
            "chunking_version": ["v1", "v1", "v1"],
            "metadata_json": ["{}", "{}", "{}"],
        }
    )
    manifest = {
        "model_id": "Qwen/Qwen3-Embedding-0.6B",
        "embedding_dim": 1024,
        "dtype": "float32",
        "input_table": "chunks.parquet",
        "chunking_version": "v1",
        "normalizer_version": "v1",
        "created_at": "2026-01-01T00:00:00Z",
        "n_chunks": manifest_n_chunks,
    }

    documents.to_parquet(documents_path)
    chunks.to_parquet(chunks_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return documents_path, chunks_path, manifest_path


def test_loaders_read_valid_parquet(tmp_path: Path) -> None:
    documents_path, chunks_path, _ = _write_populares_files(tmp_path)

    assert len(load_populares_documents(documents_path)) == 2
    assert len(load_populares_chunks(chunks_path)) == 3


def test_summary_counts_are_json_serializable(tmp_path: Path) -> None:
    documents_path, chunks_path, manifest_path = _write_populares_files(tmp_path)

    summary = summarize_populares_inputs(documents_path, chunks_path, manifest_path)

    assert summary["n_documents"] == 2
    assert summary["n_chunks"] == 3
    assert summary["n_sources"] == 1
    assert summary["source_ids"] == ["uls"]
    assert summary["section_type_counts_documents"] == {
        "campo_laboral": 1,
        "perfil_egreso": 1,
    }
    assert summary["section_type_counts_chunks"] == {
        "campo_laboral": 1,
        "perfil_egreso": 2,
    }
    assert summary["manifest_chunk_count_matches_actual"] is True
    json.dumps(summary)


def test_summary_reports_nonfatal_data_quality_edges(tmp_path: Path) -> None:
    documents_path, chunks_path, manifest_path = _write_populares_files(
        tmp_path,
        duplicate_hash=True,
        empty_text=True,
        unknown_doc_id=True,
        manifest_n_chunks=99,
    )

    summary = summarize_populares_inputs(documents_path, chunks_path, manifest_path)

    assert summary["empty_document_text_count"] == 1
    assert summary["empty_chunk_text_count"] == 1
    assert summary["duplicate_document_content_sha256_count"] == 1
    assert summary["duplicate_chunk_content_sha256_count"] == 1
    assert summary["chunks_without_matching_document_count"] == 1
    assert summary["manifest_n_chunks"] == 99
    assert summary["manifest_chunk_count_matches_actual"] is False


def test_cli_populares_validate_prints_summary(tmp_path: Path) -> None:
    documents_path, chunks_path, manifest_path = _write_populares_files(tmp_path)

    result = CliRunner().invoke(
        main,
        [
            "populares-validate",
            "--documents",
            str(documents_path),
            "--chunks",
            str(chunks_path),
            "--manifest",
            str(manifest_path),
        ],
    )

    assert result.exit_code == 0, result.output
    summary = json.loads(result.output)
    assert summary["n_documents"] == 2
    assert summary["manifest_model_id"] == "Qwen/Qwen3-Embedding-0.6B"


def test_build_populares_gold_writes_apolo_rag_contract(tmp_path: Path) -> None:
    documents_path, chunks_path, manifest_path = _write_populares_files(tmp_path)
    out_dir = tmp_path / "gold"

    manifest = build_populares_gold(
        documents_path,
        chunks_path,
        out_dir,
        manifest_path,
    )

    retrieval = pd.read_parquet(out_dir / "retrieval_corpus.parquet")
    skill_share = pd.read_parquet(out_dir / "skill_share_by_period.parquet")
    written_manifest = json.loads(
        (out_dir / "dataset_manifest.json").read_text(encoding="utf-8")
    )

    assert list(retrieval.columns) == [
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
    assert len(retrieval) == 3
    assert retrieval.loc[0, "source_name"] == "Universidad de La Serena"
    assert set(skill_share.columns) == {
        "period",
        "skill_name",
        "skill_family",
        "n_mentions",
        "n_documents",
        "share",
        "source_type",
        "source_id",
        "computed_at",
    }
    assert {"python", "sql", "data analysis", "project management"}.issubset(
        set(skill_share["skill_name"])
    )
    assert manifest == written_manifest
    assert written_manifest["gold_contract_version"] == "0.0.1-draft"
    assert written_manifest["producer"] == "vectorjobs"
    assert written_manifest["n_chunks"] == 3
    assert written_manifest["n_documents"] == 2
    assert written_manifest["embedding_model_id"] is None
    assert written_manifest["embedding_dim"] is None
    assert written_manifest["index_type"] == "lexical"


def test_build_populares_gold_fails_when_chunk_doc_is_missing(tmp_path: Path) -> None:
    documents_path, chunks_path, _ = _write_populares_files(
        tmp_path,
        unknown_doc_id=True,
    )

    with pytest.raises(ValueError, match="chunks.parquet references missing documents"):
        build_populares_gold(documents_path, chunks_path, tmp_path / "gold")


def test_cli_populares_build_gold_prints_manifest(tmp_path: Path) -> None:
    documents_path, chunks_path, _ = _write_populares_files(tmp_path)
    out_dir = tmp_path / "gold"

    result = CliRunner().invoke(
        main,
        [
            "populares-build-gold",
            "--documents",
            str(documents_path),
            "--chunks",
            str(chunks_path),
            "--out",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    manifest = json.loads(result.output)
    assert manifest["retrieval_corpus"] == "retrieval_corpus.parquet"
    assert (out_dir / "skill_share_by_period.parquet").exists()
