"""Optional real embedding backends for local evaluation runs."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np


_MISSING_SENTENCE_TRANSFORMERS = (
    "sentence-transformers is required for backend='sentence-transformer'"
)


class SentenceTransformerBackend:
    """SentenceTransformer wrapper loaded only when explicitly requested."""

    name = "sentence-transformer"

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        batch_size: int | None = None,
    ) -> None:
        if not model_name:
            raise ValueError("model_name is required for backend='sentence-transformer'")

        sentence_transformers = _import_sentence_transformers()
        model_cls = sentence_transformers.SentenceTransformer
        kwargs: dict[str, Any] = {}
        if device != "auto":
            kwargs["device"] = device

        self._model = model_cls(model_name, **kwargs)
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.embedding_dim = self._embedding_dim()

    def encode(self, texts: list[str]) -> np.ndarray:
        kwargs: dict[str, Any] = {
            "convert_to_numpy": True,
            "show_progress_bar": False,
        }
        if self.batch_size is not None:
            kwargs["batch_size"] = self.batch_size
        vectors = self._model.encode(texts, **kwargs)
        return np.asarray(vectors, dtype=np.float32)

    def _embedding_dim(self) -> int | None:
        getter = getattr(self._model, "get_sentence_embedding_dimension", None)
        if getter is None:
            return None
        dim = getter()
        return None if dim is None else int(dim)


def _import_sentence_transformers() -> Any:
    try:
        return importlib.import_module("sentence_transformers")
    except ImportError as exc:
        raise RuntimeError(_MISSING_SENTENCE_TRANSFORMERS) from exc
