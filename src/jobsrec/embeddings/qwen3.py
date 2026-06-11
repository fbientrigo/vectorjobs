"""
Qwen3 embedding backend implementation using sentence-transformers.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from jobsrec.embeddings.base import EmbeddingBackend

logger = logging.getLogger(__name__)


class Qwen3EmbeddingBackend(EmbeddingBackend):
    """
    Qwen3 embedding backend using SentenceTransformers.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        device: str = "auto",
        normalize_embeddings: bool = True,
    ) -> None:
        """
        Initialize the Qwen3 embedding backend.

        Parameters
        ----------
        model_name:
            The HuggingFace model name to load.
        device:
            Device to load the model on: 'auto', 'cpu', or 'cuda'.
        normalize_embeddings:
            If True, embeddings are normalized (cosine similarity becomes dot product).
        """
        self._model_name = model_name
        self._device = device if device != "auto" else None
        self._normalize = normalize_embeddings

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "Qwen3 embedding backend requires the `sentence-transformers` package. "
                "Install it with: pip install sentence-transformers"
            ) from e

        logger.info(f"Loading Qwen3 model: {model_name} on device: {device}")
        
        # We rely on sentence-transformers doing the heavy lifting.
        self._model = SentenceTransformer(
            model_name_or_path=model_name,
            device=self._device,
            trust_remote_code=True,
        )

    @property
    def backend_name(self) -> str:
        return "qwen3"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dim(self) -> Optional[int]:
        if hasattr(self._model, "get_sentence_embedding_dimension"):
            return self._model.get_sentence_embedding_dimension()
        return None

    def encode_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode a list of texts into a dense numpy array.

        Parameters
        ----------
        texts:
            A list of strings to encode.
        batch_size:
            Batch size to use during encoding.

        Returns
        -------
        numpy.ndarray
            A 2D array of shape (len(texts), embedding_dim).

        Raises
        ------
        TypeError
            If *texts* is a single string instead of a list.
        """
        if isinstance(texts, str):
            raise TypeError("texts must be a list of strings, not a single string.")
        if not isinstance(texts, list):
            raise TypeError(f"texts must be list[str], got {type(texts).__name__}")

        if not texts:
            return np.array([])

        logger.info(f"Encoding {len(texts)} texts using Qwen3 (batch_size={batch_size})")
        
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )
        return np.asarray(embeddings)
