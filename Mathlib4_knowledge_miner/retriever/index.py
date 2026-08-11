"""
retriever/index.py
==================

LemmaIndex — unified loader for the dual-GNN lemma embedding artifact.

This is the canonical entry point for loading the lemma corpus artifacts
supplied by the mentor.  It loads:

    lemma_vectors.npy   — (N, 512) float32, pre-normalised to unit L2 norm
    lemma_ids.json      — list[int], length N; row i → lemma_id
    lemma_hnsw.index    — hnswlib cosine-space index (optional, rebuilt if absent)
    faiss.index         — FAISS IndexFlatIP (optional, used by FAISSRetriever)
    manifest.json       — build metadata (normalize flag, dimension, count)
    lemmas.jsonl        — corpus: {lemma_id, name, statement, ...}

Design
------
LemmaIndex itself is independent of the retrieval algorithm.  The semantic
and hybrid retrievers accept a LemmaIndex as their data source.

This separation makes it easy to swap the graph backend (JSON → Neo4j)
or the ANN index (hnswlib → FAISS → ScaNN) without touching retrieval logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np


class LemmaIndex:
    """Unified loader for the lemma embedding artifact.

    Parameters
    ----------
    vectors : np.ndarray, shape (N, D), dtype float32
        Pre-normalised lemma embedding matrix.
    lemma_ids : list[int], length N
        Maps row index → lemma_id.
    corpus : dict[int, dict], keyed by lemma_id
        Full metadata for each lemma: name, statement, namespace, module.
    dimension : int
        Embedding dimensionality (D).  Inferred from vectors if not supplied.
    normalize : bool
        Whether embeddings were normalised at build time (inner product = cosine).
    """

    def __init__(
        self,
        vectors: np.ndarray,
        lemma_ids: list[int],
        corpus: dict[int, dict],
        *,
        dimension: Optional[int] = None,
        normalize: bool = True,
    ) -> None:
        self.vectors   = vectors
        self.lemma_ids = lemma_ids
        self.corpus    = corpus
        self.normalize = normalize

        self.count     = len(lemma_ids)
        self.dimension = dimension if dimension is not None else vectors.shape[1]

        if vectors.shape[0] != self.count:
            raise ValueError(
                f"Vector count ({vectors.shape[0]}) does not match "
                f"lemma_ids count ({self.count})"
            )

        # Row ↔ lemma_id mappings (O(1) lookup in both directions)
        self.row_to_id: dict[int, int] = {
            row: lid for row, lid in enumerate(lemma_ids)
        }
        self.id_to_row: dict[int, int] = {
            lid: row for row, lid in enumerate(lemma_ids)
        }

    # ------------------------------------------------------------------
    # Corpus access
    # ------------------------------------------------------------------

    def get_name(self, lemma_id: int) -> Optional[str]:
        """Return the declaration name for a lemma ID, or None."""
        entry = self.corpus.get(int(lemma_id))
        return entry["name"] if entry else None

    def get_meta(self, lemma_id: int) -> Optional[dict]:
        """Return the full metadata dict for a lemma ID, or None."""
        return self.corpus.get(int(lemma_id))

    def get_vector(self, lemma_id: int) -> Optional[np.ndarray]:
        """Return the embedding vector for a lemma ID, or None."""
        row = self.id_to_row.get(int(lemma_id))
        if row is None:
            return None
        return self.vectors[row]

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        index_dir: str | Path,
        corpus_path: Optional[str | Path] = None,
    ) -> "LemmaIndex":
        """Load from the standard artifact directory layout.

        Expected files inside ``index_dir``:
            faiss.index
            lemma_ids.json
            lemma_vectors.npy
            manifest.json   (optional — used for ``normalize`` flag)

        Parameters
        ----------
        index_dir : str | Path
            Directory containing the index artifacts.
        corpus_path : str | Path | None
            Path to ``lemmas.jsonl``.  If None, corpus is empty (no names/statements).
        """
        index_dir = Path(index_dir)

        ids_path     = index_dir / "lemma_ids.json"
        vectors_path = index_dir / "lemma_vectors.npy"
        manifest_path = index_dir / "manifest.json"

        for p in (ids_path, vectors_path):
            if not p.exists():
                raise FileNotFoundError(f"LemmaIndex: required file missing: {p}")

        # -- lemma IDs --
        lemma_ids = [int(x) for x in json.loads(ids_path.read_text("utf-8"))]

        # -- vectors --
        vectors = np.load(str(vectors_path), mmap_mode="r").astype(np.float32)

        # -- manifest --
        normalize = True
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text("utf-8"))
            normalize = bool(manifest.get("normalize", True))

        # -- corpus --
        corpus: dict[int, dict] = {}
        if corpus_path is not None:
            corpus_path = Path(corpus_path)
            if corpus_path.exists():
                with open(corpus_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            obj = json.loads(line)
                            lid = int(obj["lemma_id"])
                            corpus[lid] = obj

        return cls(
            vectors=vectors,
            lemma_ids=lemma_ids,
            corpus=corpus,
            normalize=normalize,
        )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"LemmaIndex("
            f"count={self.count}, "
            f"dim={self.dimension}, "
            f"normalize={self.normalize}, "
            f"corpus_size={len(self.corpus)})"
        )
