from pathlib import Path

from apolo_eval.datasets import load_pairs, load_triplets
from apolo_eval.runner import run
from apolo_eval.backends import MockEmbeddingBackend


FIXTURES = Path(__file__).parent / "fixtures" / "eval"


def test_pair_fixtures_load_expected_difficulties() -> None:
    pairs = load_pairs(FIXTURES / "synthetic_job_pairs.jsonl")

    assert len(pairs) == 5
    assert {pair.difficulty for pair in pairs} >= {
        "easy_positive",
        "hard_positive",
        "easy_negative",
        "hard_negative",
        "spanish_english_positive",
    }
    assert {pair.label for pair in pairs} == {0, 1}


def test_triplet_fixtures_load_expected_fields() -> None:
    triplets = load_triplets(FIXTURES / "synthetic_job_triplets.jsonl")

    assert len(triplets) == 5
    assert all(item.anchor and item.positive and item.negative for item in triplets)


def test_runner_creates_report(tmp_path: Path) -> None:
    out = tmp_path / "job_understanding_mock.md"

    result = run(
        FIXTURES / "synthetic_job_triplets.jsonl",
        MockEmbeddingBackend(),
        out,
    )

    text = out.read_text(encoding="utf-8")
    assert result["backend"] == "mock"
    assert "Recall@K" in text
    assert "MRR@K" in text
    assert "nDCG@K" in text
    assert "Hard-negative error rate" in text
    assert "Number of examples: 5" in text
    assert "Backend name: mock" in text
