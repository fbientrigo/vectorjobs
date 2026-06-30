"""
Dense embedding storage and artifact builder.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DenseArtifacts:
    """Paths produced by dense embedding builders."""

    embeddings_path: Path
    index_path: Path
    manifest_path: Path
    n_rows: int
    embedding_dim: int


def build_and_save_dense(
    backend: Any,  # Expected EmbeddingBackend, but avoiding circular/heavy typing if unneeded
    documents: list[str],
    job_ids: list[str],
    output_dir: Path | str,
    input_path: Path | str = "",
    batch_size: int = 32,
    normalize: bool = True,
    sample_size: int | None = None,
) -> DenseArtifacts:
    """
    Encode documents using a dense backend and save artifacts.

    Parameters
    ----------
    backend:
        An instance implementing `EmbeddingBackend`.
    documents:
        List of texts to encode.
    job_ids:
        Parallel list of job IDs.
    output_dir:
        Destination directory.
    input_path:
        Original path to the data, stored in the manifest.
    batch_size:
        Batch size used for encoding.
    normalize:
        Whether embeddings are normalized (stored in manifest).
    sample_size:
        The number of samples used, if limited.

    Returns
    -------
    DenseArtifacts
    """
    import pandas as pd  # Keep pandas import inside

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(documents) != len(job_ids):
        raise ValueError("documents and job_ids must have the same length")

    logger.info(f"Encoding {len(documents)} documents using {backend.backend_name}")
    embeddings = backend.encode_texts(documents, batch_size=batch_size)

    if embeddings.shape[0] != len(job_ids):
        raise RuntimeError("Embeddings row count does not match job_ids count")

    n_rows, embedding_dim = embeddings.shape

    # Save embeddings
    embeddings_path = output_dir / "job_embeddings.npy"
    np.save(embeddings_path, embeddings)

    # Save index
    index_path = output_dir / "job_ids.parquet"
    index_df = pd.DataFrame({"job_id": job_ids})
    index_df.to_parquet(index_path, index=False)

    # Build manifest
    manifest: dict[str, Any] = {
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "backend_name": backend.backend_name,
        "model_name": backend.model_name,
        "n_rows": n_rows,
        "embedding_dim": embedding_dim,
        "normalized": normalize,
        "batch_size": batch_size,
        "text_column": "job_card_text",
        "id_column": "job_id",
    }
    if sample_size is not None:
        manifest["sample_size"] = sample_size

    manifest_path = output_dir / "embedding_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    logger.info(f"Saved dense artifacts to {output_dir}")

    return DenseArtifacts(
        embeddings_path=embeddings_path,
        index_path=index_path,
        manifest_path=manifest_path,
        n_rows=n_rows,
        embedding_dim=embedding_dim,
    )
