"""
evaluation/faiss_retriever.py
=============================

Thin wrapper around the mentor's FAISS lemma index for semantic retrieval
of proof-state embeddings.

This mirrors the reference LemmaIndex class but is self-contained:
no torch_geometric required, pure numpy + faiss.

The query is the 512-dim GNN proof-state embedding — NOT a lemma embedding.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class FAISSRetriever:
    """Load the FAISS inner-product lemma index and query with proof-state embeddings.

    The index uses IndexFlatIP (exact inner product).  Vectors in the index
    were normalized at build time (normalize=True in manifest.json), so
    inner product == cosine similarity.

    For best reproducibility the query vector must also be L2-normalized
    before searching (this matches the training setup).
    """

    def __init__(
        self,
        index,
        lemma_ids: list[int],
        lemma_vectors: np.ndarray,
        *,
        normalize_queries: bool = True,
    ) -> None:
        import faiss  # noqa: F401 — confirmed available
        self.index = index
        self.lemma_ids = lemma_ids
        self.lemma_vectors = lemma_vectors
        self.normalize_queries = normalize_queries
        self._num_lemmas = len(lemma_ids)

    @classmethod
    def load(cls, index_dir: str | Path, *, normalize_queries: bool = True) -> "FAISSRetriever":
        """Load from a directory containing faiss.index, lemma_ids.json, lemma_vectors.npy.

        Parameters
        ----------
        index_dir : path to the index directory
        normalize_queries : whether to L2-normalize query vectors before search
                            (must match training; default True per manifest.json)
        """
        import faiss

        index_dir = Path(index_dir)
        index_path   = index_dir / "faiss.index"
        ids_path     = index_dir / "lemma_ids.json"
        vectors_path = index_dir / "lemma_vectors.npy"

        missing = [p for p in (index_path, ids_path, vectors_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"FAISSRetriever: missing files: {[str(p) for p in missing]}"
            )

        # Check manifest for normalize setting
        manifest_path = index_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            normalize_queries = bool(manifest.get("normalize", normalize_queries))

        lemma_ids    = [int(x) for x in json.loads(ids_path.read_text(encoding="utf-8"))]
        lemma_vectors = np.load(str(vectors_path)).astype(np.float32)
        index        = faiss.read_index(str(index_path))

        return cls(index, lemma_ids, lemma_vectors, normalize_queries=normalize_queries)

    def search(
        self,
        query_vector: np.ndarray,
        *,
        k: int = 5000,
    ) -> list[int]:
        """Search the FAISS index with a proof-state embedding.

        Parameters
        ----------
        query_vector : np.ndarray, shape (512,) or (1, 512)
        k : number of top results to return

        Returns
        -------
        list[int]
            Ordered list of lemma IDs (length <= k).
        """
        query = np.asarray(query_vector, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        if self.normalize_queries:
            norm = np.linalg.norm(query, axis=1, keepdims=True)
            norm = np.clip(norm, 1e-12, None)
            query = query / norm

        k_clamped = min(k, self._num_lemmas)
        _scores, indices = self.index.search(query, k_clamped)

        result: list[int] = []
        for idx in indices[0]:
            idx = int(idx)
            if 0 <= idx < self._num_lemmas:
                result.append(self.lemma_ids[idx])

        return result

    def search_vector(
        self,
        vector: np.ndarray,
        k: int = 5000,
    ) -> list[int]:
        """Alias for search() — matches HNSWRetriever.search_vector() signature."""
        return self.search(vector, k=k)
