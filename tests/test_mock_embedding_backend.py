import numpy as np

from apolo_eval.backends import EmbeddingBackend, MockEmbeddingBackend


def test_mock_backend_is_deterministic() -> None:
    backend = MockEmbeddingBackend(dim=32)
    texts = ["Data engineer ETL pipelines", "Frontend React developer"]

    first = backend.encode(texts)
    second = backend.encode(texts)

    np.testing.assert_allclose(first, second)


def test_mock_backend_returns_one_vector_per_text() -> None:
    backend = MockEmbeddingBackend(dim=16)
    vectors = backend.encode(["one", "two", "three"])

    assert vectors.shape == (3, 16)


def test_mock_backend_satisfies_protocol() -> None:
    backend: EmbeddingBackend = MockEmbeddingBackend()

    assert backend.name == "mock"
    assert backend.encode(["software engineer"]).shape[0] == 1
