"""
retriever — hybrid Mathlib lemma retrieval package.

Public API
----------
    from retriever import HNSWRetriever, DependencyGraph, LemmaGraphMapper
    from retriever import LogicalTopologicalRetriever
    from retriever.models import LemmaCandidate, RetrievalConfig
    from retriever.index import LemmaIndex

Sub-modules
-----------
    retriever.semantic   — HNSW-backed semantic search
    retriever.graph      — Mathlib dependency graph traversal
    retriever.hybrid     — multi-stage hybrid retrieval + scoring
    retriever.index      — lemma-index loader (vectors + IDs + HNSW)
    retriever.models     — typed data models
"""

from retriever.semantic import HNSWRetriever       # noqa: F401
from retriever.graph    import GraphProvider        # noqa: F401
from retriever.hybrid   import HybridRetriever      # noqa: F401
from retriever.index    import LemmaIndex            # noqa: F401
from retriever.models   import LemmaCandidate, RetrievalConfig  # noqa: F401

__all__ = [
    "HNSWRetriever",
    "GraphProvider",
    "HybridRetriever",
    "LemmaIndex",
    "LemmaCandidate",
    "RetrievalConfig",
]
