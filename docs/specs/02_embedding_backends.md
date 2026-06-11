# Milestone 2: Embedding Backends

## Objective
Introduce a reusable dense embedding backend interface to the project, alongside the existing TF-IDF implementation.

## Backend Interface
All embedding backends must implement the `EmbeddingBackend` protocol/abstract base class, defined in `src/jobsrec/embeddings/base.py`.

Requirements:
- `backend_name`: String identifier (e.g. `qwen3`, `fake`)
- `model_name`: Specific model used
- `embedding_dim`: Dimensionality of vectors
- `encode_texts(texts: list[str], batch_size: int) -> np.ndarray`: Main encoding function

## TF-IDF vs Dense Embeddings
- **TF-IDF**: Sparse, memory-efficient, exact lexical matches. Returns an NxV matrix (N=documents, V=vocabulary size).
- **Dense Embeddings**: Dense semantic vectors. Returns an NxD matrix (N=documents, D=embedding dimensions).

## Artifact Contract
Building dense embeddings produces the following artifacts in the specified output directory:
- `job_embeddings.npy`: A 2D numpy array of shape (N, D).
- `job_ids.parquet`: A parquet file containing the order of `job_id` corresponding to the matrix rows.
- `embedding_manifest.json`: Metadata for reproducibility.

## Manifest Contract
The `embedding_manifest.json` contains:
- `created_at`: ISO format timestamp
- `input_path`: Source dataset path
- `output_dir`: Output artifact directory
- `backend_name`: Name of the backend used
- `model_name`: Model name used
- `n_rows`: Total number of records
- `embedding_dim`: Dimensionality
- `normalized`: Boolean indicating if vectors are L2-normalized
- `batch_size`: Batch size used
- `text_column`: Column name used for texts (`job_card_text`)
- `id_column`: Column name used for identifiers (`job_id`)
- `sample_size` (Optional): If a subsample was used

## Hardware Notes
- Dense embedding models (like Qwen3) generally benefit from GPU acceleration (CUDA, T4/L4).
- The implementation allows fallback to `device="cpu"` for testing and local generation.
- Model weights are dynamically downloaded and should not be committed to the repository.
- Unit tests use a `FakeDenseEmbeddingBackend` to avoid downloading model weights or requiring GPUs.

## Future FAISS Milestone Boundary
This milestone intentionally does not include Approximate Nearest Neighbor (ANN) search like FAISS. Cosine similarity is computed exhaustively via dot product, which is sufficient for smaller datasets but will be optimized in the upcoming FAISS milestone.
