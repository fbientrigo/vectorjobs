import pytest

from apolo_eval.metrics import (
    hard_negative_error_rate,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


def test_recall_at_k_on_hand_made_rankings() -> None:
    rankings = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    assert recall_at_k(rankings, 1) == pytest.approx(1 / 3)
    assert recall_at_k(rankings, 2) == pytest.approx(2 / 3)


def test_mrr_at_k_on_hand_made_rankings() -> None:
    rankings = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    assert mrr_at_k(rankings, 3) == pytest.approx((1.0 + 0.5 + 1 / 3) / 3)
    assert mrr_at_k(rankings, 2) == pytest.approx((1.0 + 0.5 + 0.0) / 3)


def test_ndcg_at_k_on_hand_made_rankings() -> None:
    rankings = [[1, 0, 0], [0, 1, 0]]

    expected = (1.0 + (1.0 / 1.5849625007211563)) / 2
    assert ndcg_at_k(rankings, 2) == pytest.approx(expected)


def test_hard_negative_error_rate_counts_only_hard_cases() -> None:
    positive_scores = [0.9, 0.7, 0.4]
    negative_scores = [0.2, 0.8, 0.5]
    difficulties = ["easy_positive", "hard_negative", "hard_positive"]

    assert hard_negative_error_rate(
        positive_scores,
        negative_scores,
        difficulties,
    ) == pytest.approx(1.0)
