"""
Top-k cosine-similarity retrieval over a sparse TF-IDF matrix.

Design constraints
------------------
* **No dense NxN matrix** — we compute cosine similarity only between the
  query vector(s) and the full corpus, keeping memory O(n × vocab) not
  O(n²).
* The query job itself is always excluded from results (self-match guard).
* Returns structured :class:`RetrievalResult` objects — nothing is printed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

from jobsrec.embeddings.tfidf import TfidfBackend

logger = logging.getLogger(__name__)
JobId: TypeAlias = str | int


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScoredJob:
    """A single recommendation result."""

    job_id: JobId
    score: float
    rank: int


@dataclass(frozen=True)
class RetrievalResult:
    """Top-k recommendations for a single query job."""

    query_job_id: JobId
    results: list[ScoredJob]


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class TfidfRetriever:
    """
    Cosine-similarity retriever backed by a pre-fitted :class:`TfidfBackend`.

    Parameters
    ----------
    backend:
        A fitted ``TfidfBackend`` instance (matrix must be available).
    job_ids:
        Ordered list of ``job_id`` values corresponding to matrix rows.
        Must have the same length as ``backend.matrix.shape[0]``.
    """

    def __init__(
        self,
        backend: TfidfBackend,
        job_ids: list[JobId],
    ) -> None:
        if len(job_ids) != backend.matrix.shape[0]:
            raise ValueError(
                f"job_ids length ({len(job_ids)}) must match "
                f"matrix rows ({backend.matrix.shape[0]})"
            )
        self._backend = backend
        self._job_ids = np.asarray([str(job_id) for job_id in job_ids], dtype=object)

    # ------------------------------------------------------------------

    def recommend(
        self,
        query_job_id: JobId,
        top_k: int = 10,
    ) -> RetrievalResult:
        """
        Return the top-k most similar jobs to *query_job_id*.

        The query job itself is **always excluded** from results.

        Parameters
        ----------
        query_job_id:
            The ``job_id`` to use as the search query.
        top_k:
            Maximum number of results to return.

        Returns
        -------
        RetrievalResult

        Raises
        ------
        KeyError
            If *query_job_id* is not found in the corpus.
        """
        query_idx = self._find_index(query_job_id)

        # Extract query row (keeps it sparse → O(vocab) memory)
        query_vec: sp.csr_matrix = self._backend.matrix[query_idx]

        # Compute cosine similarities against the full corpus
        # Shape: (1, n_docs)
        sims: np.ndarray = cosine_similarity(
            query_vec, self._backend.matrix
        ).ravel()

        # Exclude self-match
        sims[query_idx] = -1.0

        # Partial sort — avoids a full sort for large n
        n_candidates = min(top_k, len(sims) - 1)
        if n_candidates <= 0:
            return RetrievalResult(query_job_id=query_job_id, results=[])

        # argpartition gives the top-k indices (unsorted among themselves)
        top_indices = np.argpartition(sims, -n_candidates)[-n_candidates:]
        # Sort descending by score
        top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]

        results = [
            ScoredJob(
                job_id=self._job_ids[idx],
                score=float(sims[idx]),
                rank=rank + 1,
            )
            for rank, idx in enumerate(top_indices)
            if sims[idx] >= 0.0  # skip negative sentinels
        ]

        return RetrievalResult(query_job_id=query_job_id, results=results)

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_dir(cls, gold_dir: Path | str) -> "TfidfRetriever":
        """
        Load a :class:`TfidfRetriever` from a gold output directory.

        Expects the directory to contain:
        * ``tfidf_vectorizer.joblib``
        * ``tfidf_matrix.npz``
        * ``job_index.parquet``

        Parameters
        ----------
        gold_dir:
            Directory produced by ``build-tfidf`` CLI command.

        Returns
        -------
        TfidfRetriever
        """
        gold_dir = Path(gold_dir)
        backend = TfidfBackend.load(gold_dir)
        index_df = pd.read_parquet(gold_dir / "job_index.parquet")
        job_ids = index_df["job_id"].tolist()
        return cls(backend=backend, job_ids=job_ids)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_index(self, job_id: JobId) -> int:
        """Return the matrix row index for *job_id*, or raise ``KeyError``."""
        matches = np.where(self._job_ids == str(job_id))[0]
        if len(matches) == 0:
            raise KeyError(
                f"job_id={job_id!r} not found in the corpus "
                f"({len(self._job_ids)} jobs loaded)"
            )
        return int(matches[0])
