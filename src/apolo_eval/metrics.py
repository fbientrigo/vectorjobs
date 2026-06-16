"""Retrieval metrics for semantic job-equivalence evaluation."""

from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(rankings: Sequence[Sequence[int]], k: int) -> float:
    _validate_k(k)
    if not rankings:
        return 0.0
    hits = sum(1 for ranking in rankings if any(label > 0 for label in ranking[:k]))
    return hits / len(rankings)


def mrr_at_k(rankings: Sequence[Sequence[int]], k: int) -> float:
    _validate_k(k)
    if not rankings:
        return 0.0
    total = 0.0
    for ranking in rankings:
        for idx, label in enumerate(ranking[:k], start=1):
            if label > 0:
                total += 1.0 / idx
                break
    return total / len(rankings)


def ndcg_at_k(rankings: Sequence[Sequence[int]], k: int) -> float:
    _validate_k(k)
    if not rankings:
        return 0.0
    scores = []
    for ranking in rankings:
        dcg = _dcg(ranking[:k])
        ideal = _dcg(sorted(ranking, reverse=True)[:k])
        scores.append(0.0 if ideal == 0.0 else dcg / ideal)
    return sum(scores) / len(scores)


def hard_negative_error_rate(
    positive_scores: Sequence[float],
    negative_scores: Sequence[float],
    difficulties: Sequence[str],
) -> float:
    if not (len(positive_scores) == len(negative_scores) == len(difficulties)):
        raise ValueError("score and difficulty inputs must have the same length")

    hard_indices = [
        idx
        for idx, difficulty in enumerate(difficulties)
        if "hard" in difficulty.lower()
    ]
    if not hard_indices:
        return 0.0

    errors = sum(
        1
        for idx in hard_indices
        if negative_scores[idx] >= positive_scores[idx]
    )
    return errors / len(hard_indices)


def _dcg(labels: Sequence[int]) -> float:
    return sum(
        (2.0**label - 1.0) / math.log2(idx + 1)
        for idx, label in enumerate(labels, start=1)
    )


def _validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be positive")
