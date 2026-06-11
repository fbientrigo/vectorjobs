"""
TF-IDF vectoriser wrapper.

Wraps ``sklearn.feature_extraction.text.TfidfVectorizer`` and adds:
* Input validation (rejects a bare string — callers must pass list[str]).
* Joblib-based serialisation / deserialisation for the fitted vectoriser.
* Scipy sparse matrix persistence alongside the vectoriser.
* A manifest writer so every ``build-tfidf`` run is reproducible.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TfidfArtifacts:
    """Paths produced by :func:`fit_and_save`."""

    vectorizer_path: Path
    matrix_path: Path
    index_path: Path
    manifest_path: Path
    n_docs: int
    vocab_size: int


# ---------------------------------------------------------------------------
# TF-IDF backend
# ---------------------------------------------------------------------------

class TfidfBackend:
    """
    Thin wrapper around ``TfidfVectorizer`` with persistence helpers.

    Parameters
    ----------
    max_features:
        Upper bound on vocabulary size.
    ngram_range:
        Tuple ``(min_n, max_n)`` for n-gram extraction.
    sublinear_tf:
        If ``True``, apply ``1 + log(tf)`` term-frequency dampening.
    min_df:
        Minimum document frequency for a term to enter the vocabulary.
    """

    def __init__(
        self,
        max_features: int = 30_000,
        ngram_range: tuple[int, int] = (1, 2),
        sublinear_tf: bool = True,
        min_df: int = 2,
    ) -> None:
        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
            min_df=min_df,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"(?u)\b\w\w+\b",
        )
        self._matrix: sp.csr_matrix | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, documents: list[str]) -> sp.csr_matrix:
        """
        Fit the vectoriser on *documents* and return the TF-IDF matrix.

        Parameters
        ----------
        documents:
            A **list** of strings.  Passing a bare ``str`` raises
            ``TypeError`` to catch a common mistake.

        Returns
        -------
        scipy.sparse.csr_matrix
            Shape ``(n_docs, vocab_size)``.

        Raises
        ------
        TypeError
            If *documents* is a plain string rather than a list.
        ValueError
            If *documents* is empty.
        """
        _validate_document_list(documents)
        self._matrix = self._vectorizer.fit_transform(documents)
        logger.info(
            "Fitted TF-IDF: %d docs × %d features",
            self._matrix.shape[0],
            self._matrix.shape[1],
        )
        return self._matrix  # type: ignore[return-value]

    def transform(self, documents: list[str]) -> sp.csr_matrix:
        """
        Transform *documents* with the already-fitted vectoriser.

        Parameters
        ----------
        documents:
            A **list** of strings.

        Returns
        -------
        scipy.sparse.csr_matrix

        Raises
        ------
        TypeError
            If *documents* is a plain string.
        sklearn.exceptions.NotFittedError
            If the vectoriser has not been fitted yet.
        """
        _validate_document_list(documents)
        return self._vectorizer.transform(documents)  # type: ignore[return-value]

    @property
    def matrix(self) -> sp.csr_matrix:
        """The corpus TF-IDF matrix (available after :meth:`fit_transform`)."""
        if self._matrix is None:
            raise RuntimeError("Call fit_transform() before accessing .matrix")
        return self._matrix

    @property
    def vectorizer(self) -> TfidfVectorizer:
        """The underlying fitted ``TfidfVectorizer``."""
        return self._vectorizer

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, output_dir: Path) -> None:
        """
        Persist the fitted vectoriser and matrix to *output_dir*.

        Creates ``tfidf_vectorizer.joblib`` and ``tfidf_matrix.npz``.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._vectorizer, output_dir / "tfidf_vectorizer.joblib")
        sp.save_npz(str(output_dir / "tfidf_matrix.npz"), self.matrix)
        logger.info("Saved TF-IDF artefacts to %s", output_dir)

    @classmethod
    def load(cls, output_dir: Path) -> "TfidfBackend":
        """
        Reconstruct a :class:`TfidfBackend` from a previously saved directory.

        Parameters
        ----------
        output_dir:
            Directory produced by :meth:`save`.

        Returns
        -------
        TfidfBackend
            Ready for :meth:`transform` and retrieval.
        """
        output_dir = Path(output_dir)
        instance = cls.__new__(cls)
        instance._vectorizer = joblib.load(
            output_dir / "tfidf_vectorizer.joblib"
        )
        instance._matrix = sp.load_npz(str(output_dir / "tfidf_matrix.npz"))
        logger.info("Loaded TF-IDF artefacts from %s", output_dir)
        return instance


# ---------------------------------------------------------------------------
# High-level builder (called from CLI)
# ---------------------------------------------------------------------------

def fit_and_save(
    documents: list[str],
    job_ids: list[int],
    job_card_texts: list[str],
    output_dir: Path | str,
    input_path: Path | str = "",
    config: dict[str, Any] | None = None,
) -> TfidfArtifacts:
    """
    Fit a :class:`TfidfBackend`, persist all artefacts, write a manifest.

    Parameters
    ----------
    documents:
        Corpus of ``job_card_text`` strings — one per job.
    job_ids:
        Parallel list of ``job_id`` values (same order as *documents*).
    job_card_texts:
        Same as *documents* (kept for index Parquet to avoid re-reading).
    output_dir:
        Destination directory.
    config:
        Config dict stored verbatim in the manifest.

    Returns
    -------
    TfidfArtifacts
    """
    import pandas as pd  # local import — keeps module importable without pandas

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = config or {}
    tfidf_cfg = cfg.get("tfidf", {})

    ngram_raw = tfidf_cfg.get("ngram_range", [1, 2])
    backend = TfidfBackend(
        max_features=int(tfidf_cfg.get("max_features", 30_000)),
        ngram_range=(ngram_raw[0], ngram_raw[1]),
        sublinear_tf=bool(tfidf_cfg.get("sublinear_tf", True)),
        min_df=int(tfidf_cfg.get("min_df", 2)),
    )

    backend.fit_transform(documents)
    backend.save(output_dir)

    # Write job index parquet (row order = matrix row order)
    index_df = pd.DataFrame(
        {"job_id": job_ids, "job_card_text": job_card_texts}
    )
    index_path = output_dir / "job_index.parquet"
    index_df.to_parquet(index_path, index=False)

    manifest = {
        "stage": "build-tfidf",
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "n_docs": len(documents),
        "vocab_size": len(backend.vectorizer.vocabulary_),
        "input_path": str(input_path) if input_path else "",
        "output_dir": str(output_dir),
        "config": cfg,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return TfidfArtifacts(
        vectorizer_path=output_dir / "tfidf_vectorizer.joblib",
        matrix_path=output_dir / "tfidf_matrix.npz",
        index_path=index_path,
        manifest_path=manifest_path,
        n_docs=len(documents),
        vocab_size=len(backend.vectorizer.vocabulary_),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_document_list(documents: Any) -> None:
    """Raise ``TypeError`` if *documents* is not a non-empty list of strings."""
    if isinstance(documents, str):
        raise TypeError(
            "documents must be a list[str], not a bare str. "
            "Wrap it: fit_transform([your_string])"
        )
    if not isinstance(documents, list):
        raise TypeError(
            f"documents must be list[str], got {type(documents).__name__!r}"
        )
    if len(documents) == 0:
        raise ValueError("documents list must not be empty")
