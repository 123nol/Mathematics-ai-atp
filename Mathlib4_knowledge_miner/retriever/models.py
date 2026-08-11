"""
retriever/models.py
===================

Typed data models for the hybrid Mathlib lemma retriever.

These classes are pure data containers — they have no dependencies on
HNSW, FAISS, or the Mathlib graph.  They make the retriever API typed,
self-documenting, and easy to test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Per-candidate result
# ---------------------------------------------------------------------------

@dataclass
class LemmaCandidate:
    """A single retrieved lemma candidate with its scoring breakdown.

    Attributes
    ----------
    lemma_id : int
        Integer ID, row index into lemma_vectors.npy and lemma_ids.json.
    name : str
        Fully-qualified Mathlib declaration name, e.g. ``Nat.add_comm``.
    statement : str
        Lean 4 statement string (may be empty if corpus entry is absent).
    graph_name : str | None
        Matched declaration name in the Mathlib dependency graph, or None
        if the lemma could not be mapped to a graph node.
    semantic_score : float
        Cosine similarity in [0, 1] from the HNSW stage.
        Higher is more semantically similar.
    logical_score : float
        Accumulated logical support from dependency-graph expansion.
        Weighted sum of (seed_semantic_score × hop_decay) over all
        graph-expanded paths that reach this candidate.
    frequency : int
        How many distinct HNSW seed nodes expanded into this candidate.
        Higher frequency means the candidate is reachable from many
        semantically similar declarations — a topological relevance signal.
    frequency_score : float
        Normalised frequency: frequency / max_frequency across all candidates.
    best_hop : int | None
        Shortest dependency-graph path (in hops) from any HNSW seed to this
        candidate.  None if the candidate was not reached by graph expansion.
    topological_score : float
        Score derived from best_hop: 1/(best_hop+1) for hop≥1, 1.0 for hop=0.
    final_score : float
        Weighted combination of the above component scores.
        See RetrievalConfig for the weight parameters.
    supporting_seeds : list[int]
        Lemma IDs of the HNSW seed nodes from which this candidate was
        discovered via graph expansion.
    distance : float | None
        Raw HNSW cosine distance (≈ 1 − semantic_score).  None for
        graph-expansion-only candidates.
    """

    lemma_id: int
    name: str
    statement: str = ""
    graph_name: Optional[str] = None

    # Scoring breakdown
    semantic_score: float = 0.0
    logical_score: float = 0.0
    frequency: int = 0
    frequency_score: float = 0.0
    best_hop: Optional[int] = None
    topological_score: float = 0.0
    final_score: float = 0.0

    # Provenance
    supporting_seeds: list[int] = field(default_factory=list)
    distance: Optional[float] = None

    def to_dict(self) -> dict:
        """Serialise to a JSON-friendly dict."""
        return {
            "lemma_id":         self.lemma_id,
            "name":             self.name,
            "statement":        self.statement,
            "graph_name":       self.graph_name,
            "semantic_score":   self.semantic_score,
            "logical_score":    self.logical_score,
            "frequency":        self.frequency,
            "frequency_score":  self.frequency_score,
            "best_hop":         self.best_hop,
            "topological_score":self.topological_score,
            "final_score":      self.final_score,
            "supporting_seeds": self.supporting_seeds,
            "distance":         self.distance,
        }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RetrievalConfig:
    """Configurable parameters for the hybrid retrieval pipeline.

    Hybrid scoring formula
    ----------------------
    The final score for each candidate is::

        final_score =
            semantic_weight   * semantic_score
          + logical_weight    * logical_score
          + frequency_weight  * frequency_score
          + hop_weight        * topological_score

    All weights are non-negative floats; they need not sum to 1.

    Attributes
    ----------
    semantic_k : int
        Number of HNSW approximate nearest neighbours retrieved in the
        semantic stage.  These become the seed nodes for graph expansion.
    max_hops : int
        Maximum BFS depth when expanding through the dependency graph.
        hop=0 = direct HNSW hit, hop=1 = one-hop dependency, etc.
    final_k : int
        Maximum number of candidates to return after hybrid reranking.
    semantic_weight : float
        Weight of the HNSW cosine similarity score.
    logical_weight : float
        Weight of the graph-propagated logical support.
    frequency_weight : float
        Weight of the candidate reachability frequency.
    hop_weight : float
        Weight of the inverse-hop topological proximity.
    ef_search : int
        hnswlib ``ef`` parameter during search (larger = better recall,
        slower).  Set to -1 to use the index default.
    """

    semantic_k:       int   = 100
    max_hops:         int   = 2
    final_k:          int   = 20
    semantic_weight:  float = 1.0
    logical_weight:   float = 0.75
    frequency_weight: float = 0.50
    hop_weight:       float = 0.25
    ef_search:        int   = -1

    def validate(self) -> None:
        """Raise ValueError if any parameter is out of range."""
        if self.semantic_k <= 0:
            raise ValueError("semantic_k must be positive")
        if self.max_hops < 0:
            raise ValueError("max_hops must be >= 0")
        if self.final_k <= 0:
            raise ValueError("final_k must be positive")
        for name in ("semantic_weight", "logical_weight", "frequency_weight", "hop_weight"):
            v = getattr(self, name)
            if v < 0:
                raise ValueError(f"{name} must be >= 0")
