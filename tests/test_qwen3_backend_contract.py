import numpy as np
import pytest

from jobsrec.embeddings.base import EmbeddingBackend

class FakeDenseEmbeddingBackend(EmbeddingBackend):
    def __init__(self, model_name: str = "fake-model", dim: int = 4):
        self._model_name = model_name
        self._dim = dim

    @property
    def backend_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def encode_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if isinstance(texts, str):
            raise TypeError("texts must be a list of strings, not a single string.")
        if not isinstance(texts, list):
            raise TypeError(f"texts must be list[str], got {type(texts).__name__}")
            
        n_docs = len(texts)
        if n_docs == 0:
            return np.empty((0, self._dim))
            
        # Return deterministic but pseudo-random looking embeddings
        # For tests, we'll just normalize them directly.
        rng = np.random.default_rng(42)
        embeddings = rng.random((n_docs, self._dim))
        
        # Add some variation based on string length to make it deterministic but varied
        for i, t in enumerate(texts):
            embeddings[i, 0] += len(t) * 0.1
            
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms


def test_fake_backend_contract_accepts_list():
    backend = FakeDenseEmbeddingBackend()
    texts = ["hello world", "foo bar"]
    emb = backend.encode_texts(texts)
    
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (2, 4)
    # Check normalization
    norms = np.linalg.norm(emb, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-5)


def test_fake_backend_contract_rejects_string():
    backend = FakeDenseEmbeddingBackend()
    with pytest.raises(TypeError, match="texts must be a list of strings"):
        backend.encode_texts("hello world") # type: ignore
