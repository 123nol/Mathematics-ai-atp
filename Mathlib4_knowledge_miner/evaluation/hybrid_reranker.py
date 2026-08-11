"""
evaluation/hybrid_reranker.py
=============================

Hybrid reranker: FAISS semantic candidates + Mathlib dependency/topological
reranking.

This reuses the existing LogicalTopologicalRetriever architecture from
logical_topological_retriever.py without replacing it. It adapts the
interface to accept an arbitrary query vector (the GNN proof-state embedding)
rather than a lemma row index, which is required for the benchmark.

The hybrid pipeline:

    proof-state embedding (512-d)
            ↓
    FAISS semantic search → top-K candidates
            ↓
    lemma → Mathlib graph-node mapping
            ↓
    BFS expansion through dependency graph
            ↓
    final_score = sem_weight * semantic_score
                + log_weight * logical_score
                + freq_weight * frequency_score
                + hop_weight * hop_score
            ↓
    ranked candidate list (lemma IDs)

No torch_geometric required.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from pathlib import Path

import numpy as np


class HybridReranker:
    """Combine FAISS semantic retrieval with Mathlib dependency graph scoring.

    Parameters
    ----------
    faiss_retriever : FAISSRetriever
        Semantic retrieval component.
    lemma_to_graph : dict[int, str]
        Maps lemma_id → Mathlib declaration name.
    graph_to_lemma : dict[str, int]
        Maps Mathlib declaration name → lemma_id.
    outgoing : dict[str, set[str]]
        Dependency graph adjacency (declaration → set of dependencies).
    semantic_weight : float
    logical_weight : float
    frequency_weight : float
    hop_weight : float
    max_hops : int
    """

    def __init__(
        self,
        faiss_retriever,
        lemma_to_graph: dict[int, str],
        graph_to_lemma: dict[str, int],
        outgoing: dict[str, set[str]],
        *,
        semantic_weight: float = 1.0,
        logical_weight: float = 0.75,
        frequency_weight: float = 0.50,
        hop_weight: float = 0.25,
        max_hops: int = 2,
    ) -> None:
        self.faiss = faiss_retriever
        self.lemma_to_graph = lemma_to_graph
        self.graph_to_lemma = graph_to_lemma
        self.outgoing = outgoing
        self.semantic_weight = semantic_weight
        self.logical_weight = logical_weight
        self.frequency_weight = frequency_weight
        self.hop_weight = hop_weight
        self.max_hops = max_hops

    @classmethod
    def load(
        cls,
        index_dir: str | Path,
        mapping_path: str | Path,
        graph_path: str | Path,
        *,
        semantic_weight: float = 1.0,
        logical_weight: float = 0.75,
        frequency_weight: float = 0.50,
        hop_weight: float = 0.25,
        max_hops: int = 2,
    ) -> "HybridReranker":
        """Build a HybridReranker from paths.

        Parameters
        ----------
        index_dir    : directory with faiss.index, lemma_ids.json, lemma_vectors.npy
        mapping_path : lemma_graph_mapping.json
        graph_path   : mathlib_dependencies.json
        """
        from evaluation.faiss_retriever import FAISSRetriever

        faiss_retriever = FAISSRetriever.load(index_dir)

        lemma_to_graph, graph_to_lemma = _load_mapping(mapping_path)
        outgoing = _load_graph(graph_path)

        return cls(
            faiss_retriever,
            lemma_to_graph,
            graph_to_lemma,
            outgoing,
            semantic_weight=semantic_weight,
            logical_weight=logical_weight,
            frequency_weight=frequency_weight,
            hop_weight=hop_weight,
            max_hops=max_hops,
        )

    # ------------------------------------------------------------------
    # Semantic-only retrieval
    # ------------------------------------------------------------------

    def semantic_retrieve(self, query_vector: np.ndarray, *, k: int = 5000) -> list[int]:
        """Return top-k lemma IDs by semantic similarity.

        This is the BASELINE — pure FAISS inner product search.
        The query must be the GNN proof-state embedding.
        """
        return self.faiss.search(query_vector, k=k)

    # ------------------------------------------------------------------
    # Hybrid retrieval
    # ------------------------------------------------------------------

    def hybrid_retrieve(
        self,
        query_vector: np.ndarray,
        *,
        semantic_k: int = 1000,
        final_k: int = 5000,
    ) -> list[int]:
        """Return lemma IDs reranked by the hybrid score.

        Parameters
        ----------
        query_vector : np.ndarray, shape (512,)
        semantic_k   : number of FAISS candidates to use as seeds for graph expansion
        final_k      : number of final ranked results to return

        Returns
        -------
        list[int]
            Ranked lemma IDs (most relevant first), length <= final_k.
        """
        # 1. Semantic stage
        query = np.asarray(query_vector, dtype=np.float32)
        if self.faiss.normalize_queries:
            norm = np.linalg.norm(query)
            if norm > 1e-12:
                query = query / norm

        fetch_k = max(semantic_k, final_k)
        k_clamped = min(fetch_k, self.faiss._num_lemmas)
        _scores, indices = self.faiss.index.search(query.reshape(1, -1), k_clamped)

        # Build (lemma_id, semantic_score) pairs from semantic stage
        semantic_results_all: list[tuple[int, float]] = []
        for idx, score in zip(indices[0], _scores[0]):
            idx = int(idx)
            if 0 <= idx < self.faiss._num_lemmas:
                lemma_id = self.faiss.lemma_ids[idx]
                semantic_results_all.append((lemma_id, float(score)))

        semantic_seeds = semantic_results_all[:semantic_k]

        # 2. Graph expansion + scoring
        # candidate_scores is keyed by graph_name (str)
        candidate_scores: dict[str, dict] = defaultdict(
            lambda: {
                "semantic_score": 0.0,
                "frequency": 0,
                "logical_score": 0.0,
                "best_hop": math.inf,
            }
        )

        # Track semantic candidates that have no graph mapping (by lemma_id).
        # These MUST NOT be dropped — they keep their semantic score.
        unmapped_semantic: dict[int, float] = {}  # lemma_id → semantic_score
        
        # Populate unmapped_semantic with ALL semantic hits initially.
        for lemma_id, sem_score in semantic_results_all:
            unmapped_semantic[lemma_id] = sem_score

        # Seed: direct semantic hits
        for lemma_id, sem_score in semantic_seeds:
            graph_name = self.lemma_to_graph.get(lemma_id)
            if graph_name is None:
                continue
            entry = candidate_scores[graph_name]
            entry["semantic_score"] = max(entry["semantic_score"], sem_score)
            entry["frequency"] += 1
            entry["best_hop"] = 0

        # Expand through dependency graph
        for lemma_id, sem_score in semantic_seeds:
            graph_name = self.lemma_to_graph.get(lemma_id)
            if graph_name is None:
                continue
            neighborhood = _bfs_expand(graph_name, self.outgoing, max_hops=self.max_hops)
            for node, hop in neighborhood.items():
                if hop == 0:
                    continue
                entry = candidate_scores[node]
                entry["frequency"] += 1
                entry["best_hop"] = min(entry["best_hop"], hop)
                entry["logical_score"] += sem_score * _hop_score(hop)

        # 3. Frequency normalization
        max_freq = max(
            max((e["frequency"] for e in candidate_scores.values()), default=1),
            1,
        )

        # 4. Final scoring — graph-mapped candidates
        ranked: list[tuple[float, int]] = []
        seen_lemma_ids: set[int] = set()
        for graph_name, entry in candidate_scores.items():
            lemma_id = self.graph_to_lemma.get(graph_name)
            if lemma_id is None:
                continue
            bh = entry["best_hop"]
            hop_sc = 0.0 if bh == math.inf else _hop_score(int(bh))
            freq_sc = entry["frequency"] / max_freq
            final_score = (
                self.semantic_weight  * entry["semantic_score"]
                + self.logical_weight   * entry["logical_score"]
                + self.frequency_weight * freq_sc
                + self.hop_weight       * hop_sc
            )
            ranked.append((final_score, lemma_id))
            seen_lemma_ids.add(lemma_id)

        # 5. Merge unmapped semantic candidates at their pure semantic score
        # (no logical/frequency/hop bonus — they simply keep their FAISS score)
        for lemma_id, sem_score in unmapped_semantic.items():
            if lemma_id not in seen_lemma_ids:
                final_score = self.semantic_weight * sem_score
                ranked.append((final_score, lemma_id))

        ranked.sort(reverse=True)
        return [lid for _, lid in ranked[:final_k]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hop_score(hop: int) -> float:
    if hop == 0:
        return 1.0
    return 1.0 / (hop + 1)


def _bfs_expand(node: str, outgoing: dict[str, set[str]], *, max_hops: int) -> dict[str, int]:
    """BFS from node. Returns {node: min_hop_distance}."""
    distances: dict[str, int] = {node: 0}
    queue: deque[str] = deque([node])
    while queue:
        current = queue.popleft()
        hop = distances[current]
        if hop >= max_hops:
            continue
        for neighbor in outgoing.get(current, set()):
            if neighbor not in distances:
                distances[neighbor] = hop + 1
                queue.append(neighbor)
    return distances


def _load_mapping(mapping_path: str | Path) -> tuple[dict[int, str], dict[str, int]]:
    """Load lemma_graph_mapping.json.

    Returns (lemma_to_graph, graph_to_lemma).
    The mapping.json may use either format:
        {"mapping": {"123": "Mathlib.Foo"}}   (reference format)
        {"lemma_id": "graph_name"}             (flat format)
    """
    data = json.loads(Path(mapping_path).read_text(encoding="utf-8"))

    if "mapping" in data:
        raw = data["mapping"]
    else:
        raw = data

    lemma_to_graph: dict[int, str] = {}
    graph_to_lemma: dict[str, int] = {}
    for k, v in raw.items():
        try:
            lid = int(k)
            gname = str(v)
        except (ValueError, TypeError):
            continue
        lemma_to_graph[lid] = gname
        graph_to_lemma[gname] = lid

    return lemma_to_graph, graph_to_lemma


def _load_graph(graph_path: str | Path) -> dict[str, set[str]]:
    """Load mathlib_dependencies.json and return adjacency as outgoing dict."""
    data = json.loads(Path(graph_path).read_text(encoding="utf-8"))
    outgoing: dict[str, set[str]] = defaultdict(set)
    for edge in data.get("edges", []):
        src = edge.get("source")
        tgt = edge.get("target")
        if src and tgt and src != tgt:
            outgoing[src].add(tgt)
    return outgoing
