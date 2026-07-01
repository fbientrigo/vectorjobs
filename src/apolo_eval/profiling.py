"""Lightweight profiling helpers for embedding evaluation runs."""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from apolo_eval.backends import EmbeddingBackend


@dataclass
class EncodeProfile:
    latencies_sec: list[float] = field(default_factory=list)
    texts_encoded: int = 0

    @property
    def latency_p50_ms(self) -> float:
        return _percentile(self.latencies_sec, 50.0) * 1000.0

    @property
    def latency_p95_ms(self) -> float:
        return _percentile(self.latencies_sec, 95.0) * 1000.0

    @property
    def throughput_texts_per_sec(self) -> float:
        total = sum(self.latencies_sec)
        if total == 0.0:
            return 0.0
        return self.texts_encoded / total


def profiled_encode(
    backend: EmbeddingBackend,
    texts: list[str],
    profile: EncodeProfile,
) -> np.ndarray:
    started = time.perf_counter()
    vectors = backend.encode(texts)
    elapsed = time.perf_counter() - started
    profile.latencies_sec.append(elapsed)
    profile.texts_encoded += len(texts)
    return vectors


def reset_cuda_peak_memory() -> None:
    cuda = _cuda()
    if cuda is not None and cuda.is_available():
        cuda.reset_peak_memory_stats()


def cuda_peak_memory_mb() -> float | None:
    cuda = _cuda()
    if cuda is None or not cuda.is_available():
        return None
    return float(cuda.max_memory_allocated() / (1024 * 1024))


def _cuda() -> Any | None:
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return None
    return getattr(torch, "cuda", None)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, percentile))

