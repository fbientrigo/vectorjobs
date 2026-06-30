"""
Tests for the TF-IDF backend and top-k retrieval.

Covers:
* TfidfBackend.fit_transform rejects a bare string (TypeError).
* TfidfBackend.fit_transform accepts a list[str].
* TfidfBackend.fit_transform rejects an empty list (ValueError).
* Retrieval excludes the query job itself (self-match guard).
* top_k results are sorted descending by score.
* top_k cap is respected.
* KeyError is raised for unknown job_id.
* Scores are in [0.0, 1.0].
"""

from __future__ import annotations

import pytest
import scipy.sparse as sp

from jobsrec.embeddings.tfidf import TfidfBackend
from jobsrec.recommend.retrieval import RetrievalResult, ScoredJob, TfidfRetriever


# ---------------------------------------------------------------------------
# Synthetic corpus fixtures
# ---------------------------------------------------------------------------

CORPUS = [
    "Title: Data Engineer\nSkills: Python, SQL\nDescription: Build ETL pipelines.",
    "Title: ML Engineer\nSkills: Python, PyTorch\nDescription: Train neural networks.",
    "Title: DevOps Engineer\nSkills: Kubernetes, Docker\nDescription: Manage cloud infra.",
    "Title: Backend Engineer\nSkills: Go, gRPC\nDescription: Build microservices.",
    "Title: Data Scientist\nSkills: Python, R, Statistics\nDescription: Analyse data.",
]
JOB_IDS = ["101", "102", "103", "104", "105"]


@pytest.fixture(scope="module")
def fitted_backend() -> TfidfBackend:
    """A TfidfBackend fitted on the synthetic corpus."""
    backend = TfidfBackend(max_features=500, ngram_range=(1, 1), min_df=1)
    backend.fit_transform(CORPUS)
    return backend


@pytest.fixture(scope="module")
def retriever(fitted_backend: TfidfBackend) -> TfidfRetriever:
    return TfidfRetriever(backend=fitted_backend, job_ids=JOB_IDS)


# ---------------------------------------------------------------------------
# TfidfBackend — input validation
# ---------------------------------------------------------------------------

class TestTfidfBackendInputValidation:
    def test_rejects_bare_string(self) -> None:
        """A raw string must raise TypeError with a helpful message."""
        backend = TfidfBackend(max_features=100, min_df=1)
        with pytest.raises(TypeError, match="list\\[str\\]"):
            backend.fit_transform("this is a single string")  # type: ignore[arg-type]

    def test_rejects_empty_list(self) -> None:
        backend = TfidfBackend(max_features=100, min_df=1)
        with pytest.raises(ValueError, match="empty"):
            backend.fit_transform([])

    def test_rejects_non_list_iterable(self) -> None:
        """A generator is not a list — should raise TypeError."""
        backend = TfidfBackend(max_features=100, min_df=1)
        with pytest.raises(TypeError):
            backend.fit_transform(doc for doc in CORPUS)  # type: ignore[arg-type]

    def test_accepts_list_of_strings(self) -> None:
        backend = TfidfBackend(max_features=500, ngram_range=(1, 1), min_df=1)
        matrix = backend.fit_transform(CORPUS)
        assert sp.issparse(matrix)
        assert matrix.shape[0] == len(CORPUS)

    def test_transform_rejects_bare_string(self, fitted_backend: TfidfBackend) -> None:
        with pytest.raises(TypeError, match="list\\[str\\]"):
            fitted_backend.transform("single string")  # type: ignore[arg-type]

    def test_transform_accepts_list(self, fitted_backend: TfidfBackend) -> None:
        result = fitted_backend.transform(["Python data pipelines"])
        assert result.shape[0] == 1


# ---------------------------------------------------------------------------
# TfidfBackend — matrix properties
# ---------------------------------------------------------------------------

class TestTfidfBackendMatrix:
    def test_matrix_shape(self, fitted_backend: TfidfBackend) -> None:
        mat = fitted_backend.matrix
        assert mat.shape[0] == len(CORPUS)
        assert mat.shape[1] > 0

    def test_matrix_is_sparse(self, fitted_backend: TfidfBackend) -> None:
        assert sp.issparse(fitted_backend.matrix)

    def test_matrix_unavailable_before_fit(self) -> None:
        backend = TfidfBackend(max_features=100, min_df=1)
        with pytest.raises(RuntimeError, match="fit_transform"):
            _ = backend.matrix


# ---------------------------------------------------------------------------
# TfidfRetriever — self-match exclusion
# ---------------------------------------------------------------------------

class TestSelfMatchExclusion:
    def test_query_job_not_in_results(self, retriever: TfidfRetriever) -> None:
        for job_id in JOB_IDS:
            result = retriever.recommend(query_job_id=job_id, top_k=10)
            returned_ids = [r.job_id for r in result.results]
            assert job_id not in returned_ids, (
                f"Self-match: job_id={job_id} appeared in its own recommendations"
            )

    def test_results_never_contain_query(self, retriever: TfidfRetriever) -> None:
        """Redundant but explicit assertion for the key business rule."""
        result = retriever.recommend(query_job_id="101", top_k=100)
        assert "101" not in {r.job_id for r in result.results}


# ---------------------------------------------------------------------------
# TfidfRetriever — ranking and count
# ---------------------------------------------------------------------------

class TestRetrievalRanking:
    def test_top_k_is_respected(self, retriever: TfidfRetriever) -> None:
        result = retriever.recommend(query_job_id="101", top_k=3)
        assert len(result.results) <= 3

    def test_top_1_returns_one_result(self, retriever: TfidfRetriever) -> None:
        result = retriever.recommend(query_job_id="101", top_k=1)
        assert len(result.results) == 1

    def test_results_sorted_descending_by_score(
        self, retriever: TfidfRetriever
    ) -> None:
        result = retriever.recommend(query_job_id="101", top_k=4)
        scores = [r.score for r in result.results]
        assert scores == sorted(scores, reverse=True)

    def test_ranks_are_sequential_from_one(self, retriever: TfidfRetriever) -> None:
        result = retriever.recommend(query_job_id="101", top_k=4)
        ranks = [r.rank for r in result.results]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_scores_in_unit_interval(self, retriever: TfidfRetriever) -> None:
        result = retriever.recommend(query_job_id="101", top_k=10)
        for scored_job in result.results:
            assert 0.0 <= scored_job.score <= 1.0 + 1e-9, (
                f"Score {scored_job.score} out of [0, 1] range"
            )

    def test_query_job_id_in_result(self, retriever: TfidfRetriever) -> None:
        result = retriever.recommend(query_job_id="102", top_k=5)
        assert result.query_job_id == "102"


# ---------------------------------------------------------------------------
# TfidfRetriever — edge cases
# ---------------------------------------------------------------------------

class TestRetrievalEdgeCases:
    def test_unknown_job_id_raises_key_error(self, retriever: TfidfRetriever) -> None:
        with pytest.raises(KeyError):
            retriever.recommend(query_job_id="999_999")

    def test_corpus_of_two_jobs_no_self_match(self) -> None:
        """Minimal corpus: 2 docs, self-match must still be excluded."""
        tiny_corpus = [
            "Title: Engineer\nDescription: Build systems with Python.",
            "Title: Analyst\nDescription: Analyse data with SQL.",
        ]
        backend = TfidfBackend(max_features=100, ngram_range=(1, 1), min_df=1)
        backend.fit_transform(tiny_corpus)
        retriever = TfidfRetriever(backend=backend, job_ids=["1", "2"])

        result = retriever.recommend(query_job_id="1", top_k=10)
        assert "1" not in {r.job_id for r in result.results}
        assert len(result.results) <= 1  # only 1 other doc

    def test_mismatched_job_ids_length_raises(
        self, fitted_backend: TfidfBackend
    ) -> None:
        with pytest.raises(ValueError, match="job_ids length"):
            TfidfRetriever(backend=fitted_backend, job_ids=["1", "2"])  # too short

    def test_return_type_is_retrieval_result(self, retriever: TfidfRetriever) -> None:
        result = retriever.recommend(query_job_id="101", top_k=3)
        assert isinstance(result, RetrievalResult)
        for item in result.results:
            assert isinstance(item, ScoredJob)
