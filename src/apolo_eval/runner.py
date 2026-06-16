"""Command-line runner for the Apolo job-understanding evaluation harness."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from apolo_eval.backends import EmbeddingBackend, MockEmbeddingBackend
from apolo_eval.datasets import JobTriplet, load_triplets
from apolo_eval.metrics import (
    hard_negative_error_rate,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from apolo_eval.profiling import (
    EncodeProfile,
    cuda_peak_memory_mb,
    profiled_encode,
    reset_cuda_peak_memory,
)
from apolo_eval.real_backends import SentenceTransformerBackend


def run(
    fixture: str | Path,
    backend: EmbeddingBackend,
    out: str | Path,
    k: int = 1,
) -> dict[str, float | int | str]:
    examples = load_triplets(fixture)
    profile = EncodeProfile()
    reset_cuda_peak_memory()
    rankings, positive_scores, negative_scores = _rank_triplets(
        examples,
        backend,
        profile,
    )
    difficulties = [example.difficulty for example in examples]

    results: dict[str, float | int | str] = {
        "backend": backend.name,
        "model_name": _attr_or_not_available(backend, "model_name"),
        "device": _attr_or_not_available(backend, "device"),
        "embedding_dim": _attr_or_not_available(backend, "embedding_dim"),
        "num_examples": len(examples),
        f"recall_at_{k}": recall_at_k(rankings, k),
        f"mrr_at_{k}": mrr_at_k(rankings, k),
        f"ndcg_at_{k}": ndcg_at_k(rankings, k),
        "hard_negative_error_rate": hard_negative_error_rate(
            positive_scores,
            negative_scores,
            difficulties,
        ),
        "latency_p50_ms": profile.latency_p50_ms,
        "latency_p95_ms": profile.latency_p95_ms,
        "throughput_texts_per_sec": profile.throughput_texts_per_sec,
        "cuda_peak_memory_mb": cuda_peak_memory_mb(),
    }
    _write_report(results, out, k)
    return results


def _rank_triplets(
    examples: list[JobTriplet],
    backend: EmbeddingBackend,
    profile: EncodeProfile,
) -> tuple[list[list[int]], list[float], list[float]]:
    rankings: list[list[int]] = []
    positive_scores: list[float] = []
    negative_scores: list[float] = []

    for example in examples:
        vectors = profiled_encode(
            backend,
            [example.anchor, example.positive, example.negative],
            profile,
        )
        anchor = vectors[0]
        positive_score = _cosine(anchor, vectors[1])
        negative_score = _cosine(anchor, vectors[2])
        positive_scores.append(positive_score)
        negative_scores.append(negative_score)

        scored = sorted(
            [(positive_score, 1), (negative_score, 0)],
            key=lambda item: item[0],
            reverse=True,
        )
        rankings.append([label for _, label in scored])

    return rankings, positive_scores, negative_scores


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0.0:
        return 0.0
    return float(np.dot(left, right) / denom)


def _write_report(results: dict[str, float | int | str], out: str | Path, k: int) -> None:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Job Understanding Evaluation",
        "",
        f"- Backend name: {results['backend']}",
        f"- Model name: {results['model_name']}",
        f"- Device: {results['device']}",
        f"- Embedding dimension: {results['embedding_dim']}",
        f"- Number of examples: {results['num_examples']}",
        f"- Recall@K: {results[f'recall_at_{k}']:.4f}",
        f"- MRR@K: {results[f'mrr_at_{k}']:.4f}",
        f"- nDCG@K: {results[f'ndcg_at_{k}']:.4f}",
        f"- Hard-negative error rate: {results['hard_negative_error_rate']:.4f}",
        f"- Latency p50: {results['latency_p50_ms']:.4f} ms",
        f"- Latency p95: {results['latency_p95_ms']:.4f} ms",
        f"- Throughput: {results['throughput_texts_per_sec']:.4f} texts/sec",
        f"- CUDA peak memory: {_format_optional_float(results['cuda_peak_memory_mb'])}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _backend_from_args(args: argparse.Namespace) -> EmbeddingBackend:
    name = args.backend
    if name == "mock":
        return MockEmbeddingBackend()
    if name == "sentence-transformer":
        return SentenceTransformerBackend(
            model_name=args.model_name,
            device=args.device,
            batch_size=args.batch_size,
        )
    raise ValueError(f"unsupported backend: {name}")


def _attr_or_not_available(obj: object, name: str) -> object:
    value = getattr(obj, name, None)
    return "not_available" if value is None else value


def _format_optional_float(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.4f} MB"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    parser.add_argument(
        "--backend",
        default="mock",
        choices=["mock", "sentence-transformer"],
    )
    parser.add_argument("--model-name")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--out", required=True)
    parser.add_argument("--k", type=int, default=1)
    args = parser.parse_args()

    try:
        backend = _backend_from_args(args)
    except (RuntimeError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")

    run(args.fixture, backend, args.out, k=args.k)


if __name__ == "__main__":
    main()
