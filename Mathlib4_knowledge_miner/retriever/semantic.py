"""
retriever/semantic.py
=====================

HNSW-backed semantic lemma retrieval.

This wraps the hnswlib index built from dual-GNN lemma embeddings.
The primary query entry point accepts a raw numpy embedding vector,
making it reusable with any embedding model.

Index specification (from lemma_hnsw.json)
------------------------------------------
    space          : cosine
    dimension      : 512
    count          : 59 555
    M              : 32
    ef_construction: 200
    ef_search      : 100 (default; overridden by RetrievalConfig.ef_search)

Cosine distance vs similarity
------------------------------
hnswlib cosine space reports *distance*, defined as:

    distance ≈ 1 − cosine_similarity

Therefore semantic similarity is recovered as:

    similarity = 1.0 − distance   (clipped to [0, 1])

Normalisation
-------------
The supplied lemma_vectors.npy is pre-normalised (all norms ≈ 1.0).
Query vectors must also be L2-normalised for distance to equal 1 − cosine_sim.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np


class HNSWRetriever:
    """HNSW approximate nearest-neighbour retrieval over lemma embeddings.

    This class is intentionally thin.  It handles:
        - index loading (or rebuilding from vectors)
        - per-query search
        - row ↔ lemma_id mapping

    It does NOT know about the Mathlib graph, corpus metadata, or scoring.
    Those concerns live in :class:`retriever.hybrid.HybridRetriever`.

    Parameters
    ----------
    index_path : str | Path
        Path to the hnswlib ``.index`` file.
    ids_path : str | Path
        Path to ``lemma_ids.json``.
    vectors_path : str | Path
        Path to ``lemma_vectors.npy``.
    space : str
        hnswlib metric space.  Must match the index build space.
        Default: ``"cosine"``.
    ef_search : int
        hnswlib ``ef`` parameter.  Larger values improve recall at the cost
        of speed.  Set to -1 to use ``min(100, count)``.
    """

    def __init__(
        self,
        index_path: str | Path,
        ids_path: str | Path,
        vectors_path: str | Path,
        *,
        space: str = "cosine",
        ef_search: int = -1,
    ) -> None:
        import hnswlib

        self._index_path   = Path(index_path)
        self._ids_path     = Path(ids_path)
        self._vectors_path = Path(vectors_path)
        self._space        = space

        # -- lemma IDs --
        self.lemma_ids: list[int] = [
            int(x)
            for x in json.loads(self._ids_path.read_text("utf-8"))
        ]
        self.count     = len(self.lemma_ids)

        # Row ↔ lemma_id bi-directional maps
        self.row_to_id: dict[int, int] = {
            row: lid for row, lid in enumerate(self.lemma_ids)
        }
        self.id_to_row: dict[int, int] = {
            lid: row for row, lid in enumerate(self.lemma_ids)
        }

        # -- vectors (memory-mapped for large files) --
        self.vectors: np.ndarray = np.load(
            str(self._vectors_path), mmap_mode="r"
        )
        if self.vectors.ndim != 2:
            raise ValueError(
                f"Expected 2D vector array, got shape {self.vectors.shape}"
            )
        if self.vectors.shape[0] != self.count:
            raise ValueError(
                "Vector count does not match lemma_ids count: "
                f"{self.vectors.shape[0]} vs {self.count}"
            )
        self.dimension = self.vectors.shape[1]

        # -- HNSW index --
        self.index = hnswlib.Index(space=space, dim=self.dimension)
        self.index.load_index(str(self._index_path))

        # ef_search controls the accuracy/speed trade-off at query time.
        _ef = ef_search if ef_search > 0 else min(100, self.count)
        self.index.set_ef(_ef)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: np.ndarray,
        *,
        k: int = 20,
    ) -> list[dict]:
        """Search for the k nearest lemma embeddings.

        Parameters
        ----------
        query : np.ndarray, shape (D,) or (1, D)
            Query embedding vector.  Must be L2-normalised if the index
            uses cosine space.
        k : int
            Number of results.

        Returns
        -------
        list[dict]
            Each element has keys:
                ``row``      — index into lemma_vectors.npy
                ``lemma_id`` — mapped lemma integer ID
                ``distance`` — hnswlib cosine distance (≈ 1 − cosine_sim)
        """
        q = np.asarray(query, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if q.ndim != 2 or q.shape[1] != self.dimension:
            raise ValueError(
                f"Query shape mismatch: expected (*, {self.dimension}), got {q.shape}"
            )

        k_eff = min(int(k), self.count)
        labels, distances = self.index.knn_query(q, k=k_eff)

        return [
            {
                "row":      int(row),
                "lemma_id": self.row_to_id[int(row)],
                "distance": float(dist),
            }
            for row, dist in zip(labels[0], distances[0])
        ]

    def search_by_row(self, row: int, *, k: int = 20) -> list[dict]:
        """Use an existing lemma row as the query vector."""
        if not (0 <= row < self.count):
            raise IndexError(f"Row {row} out of range [0, {self.count})")
        return self.search(self.vectors[row], k=k)

    def search_by_id(self, lemma_id: int, *, k: int = 20) -> list[dict]:
        """Use an existing lemma ID as the query vector."""
        row = self.id_to_row.get(int(lemma_id))
        if row is None:
            raise KeyError(f"Lemma ID {lemma_id} not found in index")
        return self.search_by_row(row, k=k)

    @staticmethod
    def distance_to_similarity(distance: float) -> float:
        """Convert hnswlib cosine distance to cosine similarity in [0, 1]."""
        return max(0.0, min(1.0, 1.0 - float(distance)))

    # ------------------------------------------------------------------
    # Index construction (fallback)
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        vectors: np.ndarray,
        lemma_ids: list[int],
        output_path: str | Path,
        *,
        space: str = "cosine",
        M: int = 32,
        ef_construction: int = 200,
        ef_search: int = 100,
    ) -> None:
        """Build and save an hnswlib index from a vector matrix.

        Parameters
        ----------
        vectors : np.ndarray, shape (N, D), dtype float32
            Pre-normalised lemma embeddings.
        lemma_ids : list[int]
            Row → lemma_id mapping (must be 0..N-1 integer labels).
        output_path : str | Path
            Where to write the ``.index`` file.
        space : str
            hnswlib metric.  Use ``"cosine"`` for normalised vectors
            or ``"ip"`` (inner product) for the same result.
        M : int
            HNSW M parameter.  Controls index size and recall.
        ef_construction : int
            Build-time ef.  Higher = better recall, slower build.
        ef_search : int
            Saved as metadata comment; actual ef set when loading.
        """
        import hnswlib

        N, D = vectors.shape
        idx = hnswlib.Index(space=space, dim=D)
        idx.init_index(max_elements=N, ef_construction=ef_construction, M=M)
        idx.set_ef(ef_search)
        idx.add_items(vectors.astype(np.float32), list(range(N)))
        idx.save_index(str(output_path))

    def __repr__(self) -> str:
        return (
            f"HNSWRetriever("
            f"count={self.count}, "
            f"dim={self.dimension}, "
            f"space={self._space!r})"
        )
