"""
retriever/hybrid.py
===================

HybridRetriever — multi-stage Mathlib lemma retrieval.

Architecture
------------

::

    query embedding (512-d numpy array)
            │
            ▼
    HNSW semantic search (HNSWRetriever)
            │   top-k candidates, ranked by cosine similarity
            ▼
    lemma → Mathlib graph node mapping (LemmaGraphMapper)
            │   lemma_id → declaration name
            ▼
    BFS graph expansion (GraphProvider)
            │   multi-hop dependency traversal
            ▼
    candidate accumulation + scoring
            │   semantic, logical, frequency, topological components
            ▼
    deduplication + sorting
            │
            ▼
    final top-K LemmaCandidate list


Hybrid scoring formula
----------------------
For each candidate::

    final_score =
        semantic_weight   × semantic_score     (cosine similarity)
      + logical_weight    × logical_score      (propagated graph support)
      + frequency_weight  × frequency_score    (normalised reachability count)
      + hop_weight        × topological_score  (inverse hop distance)

where::

    semantic_score  = 1 − hnswlib_distance  (clipped to [0, 1])
    logical_score   = Σ (seed_semantic_score × hop_decay(hop))
                      over all graph-expansion paths reaching this node
    hop_decay(hop)  = 1 / (hop + 1)
    frequency_score = frequency / max_frequency_across_all_candidates
    topological_score = hop_decay(best_hop)

All weights are configurable via :class:`retriever.models.RetrievalConfig`.

Anti-leakage guarantee
-----------------------
Graph expansion starts from HNSW semantic seeds — never from gold targets.
The caller is responsible for not supplying gold-derived query embeddings.

Neo4j compatibility
-------------------
The ``graph`` parameter is typed as :class:`retriever.graph.GraphProvider`.
Replacing the JSON-backed provider with a Neo4j implementation requires no
changes to this class.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional

import numpy as np

from retriever.models import LemmaCandidate, RetrievalConfig
from retriever.graph  import GraphProvider
from retriever.semantic import HNSWRetriever


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _hop_decay(hop: int) -> float:
    """Topological proximity weight: 1.0 at hop 0, decreasing for further hops."""
    if hop == 0:
        return 1.0
    return 1.0 / (hop + 1)


# ---------------------------------------------------------------------------
# Mapper protocol
# ---------------------------------------------------------------------------

class _MapperProtocol:
    """Minimal duck-type protocol for lemma ↔ graph-name mapping.

    Implementations:
        - :class:`LemmaGraphMapper` from ``lemma_graph_mapper.py``
        - a dict wrapper (for tests)
    """

    def get_graph_name(self, lemma_id: int) -> Optional[str]:
        raise NotImplementedError

    def get_lemma(self, lemma_id: int) -> Optional[dict]:
        raise NotImplementedError


class _DictMapper:
    """Thin wrapper so dicts can be used as mappers in tests."""

    def __init__(
        self,
        lemma_to_graph: dict[int, str],
        corpus: dict[int, dict],
    ) -> None:
        self._l2g    = lemma_to_graph
        self._corpus = corpus

    def get_graph_name(self, lemma_id: int) -> Optional[str]:
        return self._l2g.get(int(lemma_id))

    def get_lemma(self, lemma_id: int) -> Optional[dict]:
        return self._corpus.get(int(lemma_id))


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """Multi-stage Mathlib lemma retriever.

    Parameters
    ----------
    hnsw : HNSWRetriever
        Semantic search component.
    graph : GraphProvider
        Mathlib dependency graph (any implementation).
    mapper
        Object with ``get_graph_name(lemma_id)`` and ``get_lemma(lemma_id)``
        methods that bridge lemma IDs ↔ graph declaration names.
    graph_to_lemma : dict[str, int]
        Reverse map: declaration name → lemma_id.
        Needed to resolve graph-expanded nodes back to lemma IDs.
    config : RetrievalConfig
        Scoring weights and search parameters.
    """

    def __init__(
        self,
        hnsw: HNSWRetriever,
        graph: GraphProvider,
        mapper,
        graph_to_lemma: dict[str, int],
        *,
        config: Optional[RetrievalConfig] = None,
    ) -> None:
        self.hnsw           = hnsw
        self.graph          = graph
        self.mapper         = mapper
        self.graph_to_lemma = graph_to_lemma
        self.config         = config or RetrievalConfig()
        self.config.validate()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def semantic_retrieve(
        self,
        query: np.ndarray,
        *,
        k: Optional[int] = None,
    ) -> list[LemmaCandidate]:
        """Semantic-only retrieval (HNSW only, no graph expansion).

        Parameters
        ----------
        query : np.ndarray, shape (512,)
            L2-normalised proof-state embedding.
        k : int | None
            Number of results.  Defaults to ``config.semantic_k``.

        Returns
        -------
        list[LemmaCandidate]
            Ranked by descending semantic score.
        """
        _k = k if k is not None else self.config.semantic_k
        raw = self.hnsw.search(query, k=_k)

        candidates: list[LemmaCandidate] = []
        for r in raw:
            lid  = int(r["lemma_id"])
            meta = self.mapper.get_lemma(lid) or {}
            sem  = HNSWRetriever.distance_to_similarity(r["distance"])
            candidates.append(LemmaCandidate(
                lemma_id       = lid,
                name           = meta.get("name", f"lemma_{lid}"),
                statement      = meta.get("statement", ""),
                graph_name     = self.mapper.get_graph_name(lid),
                semantic_score = sem,
                final_score    = sem,
                distance       = r["distance"],
            ))
        return candidates

    def retrieve(
        self,
        query: np.ndarray,
        *,
        config: Optional[RetrievalConfig] = None,
    ) -> list[LemmaCandidate]:
        """Full hybrid retrieval.

        Pipeline
        --------
        1. HNSW semantic search → top-``semantic_k`` seed candidates.
        2. Map each seed to its graph declaration name (if available).
        3. BFS-expand each mapped seed through ``max_hops`` dependency hops.
        4. Accumulate scores for all discovered nodes.
        5. Resolve graph nodes back to lemma IDs via ``graph_to_lemma``.
        6. Merge unmapped semantic candidates (retain their semantic score).
        7. Sort by ``final_score``, return top ``final_k``.

        Parameters
        ----------
        query : np.ndarray, shape (512,)
            L2-normalised proof-state embedding.
        config : RetrievalConfig | None
            Override the instance config for this call only.

        Returns
        -------
        list[LemmaCandidate]
            At most ``config.final_k`` candidates, ranked by descending
            ``final_score``.
        """
        cfg = config or self.config

        # ----------------------------------------------------------------
        # 1. Semantic stage
        # ----------------------------------------------------------------
        raw_results = self.hnsw.search(query, k=cfg.semantic_k)

        # Build (lemma_id → semantic_score) ordered list
        sem_pairs: list[tuple[int, float]] = [
            (
                int(r["lemma_id"]),
                HNSWRetriever.distance_to_similarity(r["distance"]),
            )
            for r in raw_results
        ]

        # ----------------------------------------------------------------
        # 2. Graph expansion + candidate accumulation
        # ----------------------------------------------------------------
        # Keyed by declaration name (graph namespace)
        AccEntry = dict  # semantic_score, frequency, logical_score, best_hop
        cand_by_graph: dict[str, AccEntry] = defaultdict(
            lambda: {
                "semantic_score": 0.0,
                "frequency":      0,
                "logical_score":  0.0,
                "best_hop":       math.inf,
                "seed_ids":       set(),
            }
        )

        # Track all semantic lemma IDs (for unmapped fallback)
        all_sem: dict[int, float] = {}
        for lid, sem in sem_pairs:
            all_sem[lid] = sem

        # Seed: add direct semantic candidates
        for lid, sem in sem_pairs:
            graph_name = self.mapper.get_graph_name(lid)
            if graph_name is None:
                continue
            entry = cand_by_graph[graph_name]
            entry["semantic_score"] = max(entry["semantic_score"], sem)
            entry["frequency"]     += 1
            entry["best_hop"]       = 0
            entry["seed_ids"].add(lid)

        # BFS expansion from each mapped seed
        for lid, sem in sem_pairs:
            graph_name = self.mapper.get_graph_name(lid)
            if graph_name is None:
                continue

            neighborhood = self.graph.expand(
                graph_name,
                hops=cfg.max_hops,
                direction="outgoing",
            )

            for node, hop in neighborhood.items():
                if hop == 0:
                    continue  # seed itself already handled above

                entry = cand_by_graph[node]
                entry["frequency"]  += 1
                entry["best_hop"]    = min(entry["best_hop"], hop)
                entry["logical_score"] += sem * _hop_decay(hop)
                entry["seed_ids"].add(lid)

        # ----------------------------------------------------------------
        # 3. Frequency normalisation
        # ----------------------------------------------------------------
        max_freq = max(
            (e["frequency"] for e in cand_by_graph.values()),
            default=1,
        )
        max_freq = max(max_freq, 1)

        # ----------------------------------------------------------------
        # 4. Final scoring for graph-mapped candidates
        # ----------------------------------------------------------------
        ranked: list[LemmaCandidate] = []
        seen_ids: set[int] = set()

        for graph_name, entry in cand_by_graph.items():
            lid = self.graph_to_lemma.get(graph_name)
            if lid is None:
                continue
            meta = self.mapper.get_lemma(lid) or {}

            bh = entry["best_hop"]
            hop_sc  = _hop_decay(int(bh)) if bh != math.inf else 0.0
            freq_sc = entry["frequency"] / max_freq

            final = (
                cfg.semantic_weight   * entry["semantic_score"]
                + cfg.logical_weight  * entry["logical_score"]
                + cfg.frequency_weight * freq_sc
                + cfg.hop_weight      * hop_sc
            )

            ranked.append(LemmaCandidate(
                lemma_id         = lid,
                name             = meta.get("name", f"lemma_{lid}"),
                statement        = meta.get("statement", ""),
                graph_name       = graph_name,
                semantic_score   = entry["semantic_score"],
                logical_score    = entry["logical_score"],
                frequency        = entry["frequency"],
                frequency_score  = freq_sc,
                best_hop         = None if bh == math.inf else int(bh),
                topological_score= hop_sc,
                final_score      = final,
                supporting_seeds = sorted(entry["seed_ids"]),
                distance         = (
                    None if entry["semantic_score"] == 0.0
                    else 1.0 - entry["semantic_score"]
                ),
            ))
            seen_ids.add(lid)

        # ----------------------------------------------------------------
        # 5. Merge unmapped semantic candidates (no graph bonus)
        # ----------------------------------------------------------------
        for lid, sem in all_sem.items():
            if lid in seen_ids:
                continue
            meta = self.mapper.get_lemma(lid) or {}
            final = cfg.semantic_weight * sem
            ranked.append(LemmaCandidate(
                lemma_id       = lid,
                name           = meta.get("name", f"lemma_{lid}"),
                statement      = meta.get("statement", ""),
                graph_name     = None,
                semantic_score = sem,
                final_score    = final,
                distance       = 1.0 - sem,
            ))

        # ----------------------------------------------------------------
        # 6. Sort + truncate
        # ----------------------------------------------------------------
        ranked.sort(key=lambda c: c.final_score, reverse=True)
        return ranked[: cfg.final_k]

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_paths(
        cls,
        index_path: str,
        ids_path: str,
        vectors_path: str,
        graph_path: str,
        mapping_path: str,
        corpus_path: Optional[str] = None,
        *,
        config: Optional[RetrievalConfig] = None,
    ) -> "HybridRetriever":
        """Construct a HybridRetriever from file paths.

        Parameters
        ----------
        index_path    : path to hnswlib ``.index`` file
        ids_path      : path to ``lemma_ids.json``
        vectors_path  : path to ``lemma_vectors.npy``
        graph_path    : path to ``mathlib_dependencies.json``
        mapping_path  : path to ``lemma_graph_mapping.json``
        corpus_path   : path to ``lemmas.jsonl`` (optional)
        config        : retrieval configuration
        """
        import json, sys
        from pathlib import Path

        # Import mapper from existing module (re-uses tested code)
        _repo_root = Path(__file__).resolve().parents[1]
        if str(_repo_root) not in sys.path:
            sys.path.insert(0, str(_repo_root))
        from lemma_graph_mapper import LemmaGraphMapper

        from retriever.graph import load_graph

        hnsw   = HNSWRetriever(index_path, ids_path, vectors_path)
        graph  = load_graph(graph_path)
        mapper = LemmaGraphMapper(graph_path=graph_path, lemma_path=corpus_path or "lemmas.jsonl")

        # Build reverse map: graph_name → lemma_id
        graph_to_lemma: dict[str, int] = {
            gname: lid for lid, gname in mapper.mapping.items()
        }

        return cls(hnsw, graph, mapper, graph_to_lemma, config=config)
