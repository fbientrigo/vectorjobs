"""
Abstract base class for embedding backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class EmbeddingBackend(ABC):
    """
    Protocol for dense embedding backends.
    """

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Name of the backend (e.g., 'qwen3', 'fake')."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the specific model being used."""
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> Optional[int]:
        """Dimensionality of the embeddings, if known before encoding."""
        pass

    @abstractmethod
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
        """
        pass
