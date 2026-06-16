import sys
import types

import numpy as np
import pytest

from apolo_eval.real_backends import SentenceTransformerBackend


class FakeSentenceTransformer:
    def __init__(self, model_name: str, **kwargs: object) -> None:
        self.model_name = model_name
        self.kwargs = kwargs
        self.last_encode_kwargs: dict[str, object] = {}

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        self.last_encode_kwargs = kwargs
        return np.asarray([[float(len(text)), 1.0] for text in texts])

    def get_sentence_embedding_dimension(self) -> int:
        return 2


def test_sentence_transformer_backend_uses_fake_model(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    backend = SentenceTransformerBackend(
        model_name="local-test-model",
        device="cpu",
        batch_size=4,
    )
    vectors = backend.encode(["data engineer", "software engineer"])

    assert backend.name == "sentence-transformer"
    assert backend.model_name == "local-test-model"
    assert backend.device == "cpu"
    assert backend.embedding_dim == 2
    assert vectors.shape == (2, 2)
    assert backend._model.kwargs == {"device": "cpu"}
    assert backend._model.last_encode_kwargs["batch_size"] == 4
    assert backend._model.last_encode_kwargs["convert_to_numpy"] is True


def test_sentence_transformer_backend_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "sentence_transformers", raising=False)

    def fail_import(name: str) -> object:
        if name == "sentence_transformers":
            raise ImportError(name)
        raise AssertionError(name)

    monkeypatch.setattr("importlib.import_module", fail_import)

    with pytest.raises(
        RuntimeError,
        match="sentence-transformers is required for backend='sentence-transformer'",
    ):
        SentenceTransformerBackend(model_name="missing")
