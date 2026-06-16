"""Small evaluation harness for Apolo job-understanding experiments."""

from apolo_eval.backends import EmbeddingBackend, MockEmbeddingBackend
from apolo_eval.datasets import JobPair, JobTriplet, load_pairs, load_triplets
from apolo_eval.metrics import (
    hard_negative_error_rate,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from apolo_eval.parquet_adapter import JobTextRecord, load_job_texts_from_parquet

__all__ = [
    "EmbeddingBackend",
    "JobPair",
    "JobTextRecord",
    "JobTriplet",
    "MockEmbeddingBackend",
    "hard_negative_error_rate",
    "load_job_texts_from_parquet",
    "load_pairs",
    "load_triplets",
    "mrr_at_k",
    "ndcg_at_k",
    "recall_at_k",
]
