import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jobsrec.embeddings.dense_store import build_and_save_dense
from jobsrec.recommend.dense_retrieval import DenseRetriever
from tests.test_qwen3_backend_contract import FakeDenseEmbeddingBackend


def test_build_and_save_dense(tmp_path: Path):
    backend = FakeDenseEmbeddingBackend(dim=4)
    documents = ["Software Engineer", "Data Scientist", "Product Manager"]
    job_ids = ["101", "102", "103"]
    
    artifacts = build_and_save_dense(
        backend=backend,
        documents=documents,
        job_ids=job_ids,
        output_dir=tmp_path,
        input_path="dummy.parquet"
    )
    
    assert artifacts.embeddings_path.exists()
    assert artifacts.index_path.exists()
    assert artifacts.manifest_path.exists()
    
    # Check manifest
    manifest = json.loads(artifacts.manifest_path.read_text())
    assert manifest["backend_name"] == "fake"
    assert manifest["n_rows"] == 3
    assert manifest["embedding_dim"] == 4
    assert manifest["normalized"] is True
    
    # Check embeddings
    emb = np.load(artifacts.embeddings_path)
    assert emb.shape == (3, 4)
    
    # Check index
    df = pd.read_parquet(artifacts.index_path)
    assert df["job_id"].tolist() == ["101", "102", "103"]


def test_dense_retrieval_excludes_self_match_and_finds_top_k():
    # Make perfectly known embeddings
    # row 0: [1, 0] (id=10)
    # row 1: [0, 1] (id=20)
    # row 2: [0.99, 0.14] (id=30) - very close to row 0
    # row 3: [-1, 0] (id=40) - opposite of row 0
    
    embeddings = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.989949, 0.141421], # normalized
        [-1.0, 0.0]
    ])
    job_ids = ["10", "20", "30", "40"]
    
    retriever = DenseRetriever(embeddings, job_ids)
    
    res = retriever.recommend(query_job_id="10", top_k=2)
    assert res.query_job_id == "10"
    assert len(res.results) == 2
    
    # the closest should be 30
    assert res.results[0].job_id == "30"
    assert res.results[0].rank == 1
    # next should be 20
    assert res.results[1].job_id == "20"
    assert res.results[1].rank == 2
    
    # 10 should not be in the results
    for r in res.results:
        assert r.job_id != "10"


def test_dense_retrieval_unknown_job_id_raises_keyerror():
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    job_ids = ["10", "20"]
    
    retriever = DenseRetriever(embeddings, job_ids)
    
    with pytest.raises(KeyError, match="not found in the corpus"):
        retriever.recommend(query_job_id="99")
