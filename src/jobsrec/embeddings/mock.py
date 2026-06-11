"""Deterministic lightweight embedding backend for tests and smoke runs."""

from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np

from jobsrec.embeddings.base import EmbeddingBackend


class MockEmbeddingBackend(EmbeddingBackend):
    """Small deterministic backend that does not download models."""

    def __init__(self, model_name: str = "deterministic-mock", dim: int = 16) -> None:
        self._model_name = model_name
        self._dim = dim

    @property
    def backend_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dim(self) -> Optional[int]:
        return self._dim

    def encode_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if isinstance(texts, str):
            raise TypeError("texts must be a list of strings, not a single string.")
        if not isinstance(texts, list):
            raise TypeError(f"texts must be list[str], got {type(texts).__name__}")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        vectors = np.zeros((len(texts), self._dim), dtype=np.float32)
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            for offset, text in enumerate(batch):
                digest = hashlib.sha256(str(text).encode("utf-8")).digest()
                raw = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
                tiled = np.resize(raw, self._dim)
                vector = (tiled / 127.5) - 1.0
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
                vectors[start + offset] = vector
        return vectors
