"""Embedding backends used by the evaluation harness."""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np


class EmbeddingBackend(Protocol):
    """Interface for text embedding backends."""

    name: str

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return one embedding vector per input text."""


class MockEmbeddingBackend:
    """Deterministic token-feature backend for offline tests and smoke runs."""

    name = "mock"

    def __init__(self, dim: int = 64) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.vstack([self._encode_one(text) for text in texts]).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0.0, 1.0, norms)

    def _encode_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in _tokens(text):
            idx = _stable_bucket(token, self.dim)
            vector[idx] += 1.0
        return vector


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\w+#.]+", text.lower(), flags=re.UNICODE)


def _stable_bucket(token: str, dim: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim

