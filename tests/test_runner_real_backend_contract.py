import argparse
from pathlib import Path

import numpy as np

from apolo_eval.runner import _backend_from_args, run


class ContractBackend:
    name = "sentence-transformer"
    model_name = "fake-contract-model"
    device = "cpu"
    embedding_dim = 3

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    float("engineer" in lowered),
                    float("dashboard" in lowered or "sql" in lowered),
                    float(len(text) % 7),
                ]
            )
        return np.asarray(vectors, dtype=np.float32)


def test_runner_report_includes_real_backend_metadata(tmp_path: Path) -> None:
    out = tmp_path / "report.md"

    result = run(
        "data/eval/synthetic_job_triplets.jsonl",
        ContractBackend(),
        out,
    )
    report = out.read_text(encoding="utf-8")

    assert result["backend"] == "sentence-transformer"
    assert "Backend name: sentence-transformer" in report
    assert "Model name: fake-contract-model" in report
    assert "Device: cpu" in report
    assert "Embedding dimension: 3" in report
    assert "Latency p50" in report
    assert "Latency p95" in report
    assert "Throughput" in report
    assert "CUDA peak memory" in report


def test_runner_constructs_sentence_transformer_backend_from_args(
    monkeypatch,
) -> None:
    captured = {}

    class FakeBackend:
        def __init__(self, model_name: str, device: str, batch_size: int | None) -> None:
            captured["model_name"] = model_name
            captured["device"] = device
            captured["batch_size"] = batch_size

    monkeypatch.setattr("apolo_eval.runner.SentenceTransformerBackend", FakeBackend)

    backend = _backend_from_args(
        argparse.Namespace(
            backend="sentence-transformer",
            model_name="model",
            device="cpu",
            batch_size=8,
        )
    )

    assert isinstance(backend, FakeBackend)
    assert captured == {"model_name": "model", "device": "cpu", "batch_size": 8}
