"""
Top-k retrieval over dense embeddings.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from jobsrec.recommend.retrieval import JobId, RetrievalResult, ScoredJob

logger = logging.getLogger(__name__)


class DenseRetriever:
    """
    Cosine-similarity retriever backed by dense numpy embeddings.

    Assumes embeddings are already normalized, so dot product == cosine similarity.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        job_ids: list[JobId],
    ) -> None:
        if len(job_ids) != embeddings.shape[0]:
            raise ValueError(
                f"job_ids length ({len(job_ids)}) must match "
                f"embeddings rows ({embeddings.shape[0]})"
            )
        self._embeddings = embeddings
        self._job_ids = np.asarray([str(job_id) for job_id in job_ids], dtype=object)

    def recommend(
        self,
        query_job_id: JobId,
        top_k: int = 10,
    ) -> RetrievalResult:
        """
        Return the top-k most similar jobs to *query_job_id*.

        The query job itself is excluded.
        """
        query_idx = self._find_index(query_job_id)

        # Extract query vector. Shape: (embedding_dim,)
        query_vec = self._embeddings[query_idx]

        # Compute dot product (cosine similarity since vectors are normalized)
        # Shape: (n_rows,)
        sims = np.dot(self._embeddings, query_vec)

        # Exclude self-match
        sims[query_idx] = -1.0

        n_candidates = min(top_k, len(sims) - 1)
        if n_candidates <= 0:
            return RetrievalResult(query_job_id=query_job_id, results=[])

        # argpartition for top-k
        top_indices = np.argpartition(sims, -n_candidates)[-n_candidates:]
        # sort descending
        top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]

        results = [
            ScoredJob(
                job_id=self._job_ids[idx],
                score=float(sims[idx]),
                rank=rank + 1,
            )
            for rank, idx in enumerate(top_indices)
            if sims[idx] >= 0.0
        ]

        return RetrievalResult(query_job_id=query_job_id, results=results)

    @classmethod
    def from_dir(cls, gold_dir: Path | str) -> "DenseRetriever":
        """
        Load a DenseRetriever from an output directory containing artifacts.
        """
        gold_dir = Path(gold_dir)
        embeddings_path = gold_dir / "job_embeddings.npy"
        index_path = gold_dir / "job_ids.parquet"

        logger.info(f"Loading dense embeddings from {embeddings_path}")
        embeddings = np.load(embeddings_path)
        
        index_df = pd.read_parquet(index_path)
        job_ids = index_df["job_id"].tolist()

        return cls(embeddings=embeddings, job_ids=job_ids)

    def _find_index(self, job_id: JobId) -> int:
        matches = np.where(self._job_ids == str(job_id))[0]
        if len(matches) == 0:
            raise KeyError(
                f"job_id={job_id!r} not found in the corpus "
                f"({len(self._job_ids)} jobs loaded)"
            )
        return int(matches[0])
