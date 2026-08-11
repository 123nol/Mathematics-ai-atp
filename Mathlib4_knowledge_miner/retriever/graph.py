"""
retriever/graph.py
==================

GraphProvider — abstract interface + JSON implementation for Mathlib
dependency graph traversal.

Design for future Neo4j compatibility
--------------------------------------
The :class:`GraphProvider` abstract base class defines the interface that the
hybrid retriever uses for graph lookups.  Any concrete implementation —
JSON-backed, Neo4j, SQLite, etc. — that satisfies this interface can be
plugged in without modifying the retrieval algorithm.

Current implementation
-----------------------
:class:`JsonGraphProvider` loads ``mathlib_dependencies.json``, which has
the schema::

    {
        "declarations": ["name1", "name2", ...],
        "edges": [
            {"source": "...", "target": "...", "type": "USES"},
            ...
        ]
    }

Edge semantics::

    source --USES--> target

meaning "source depends on target".  Therefore:

    outgoing(source) = declarations used by source
    incoming(target) = declarations that use target

Multi-hop expansion
-------------------
:meth:`GraphProvider.expand` performs BFS and returns a dict mapping each
reachable node to its minimum hop distance from the start node.

Anti-leakage guarantee
----------------------
Graph expansion always starts from HNSW semantic candidates or
query-related nodes — never from the gold target itself.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class GraphProvider(ABC):
    """Abstract interface for the Mathlib dependency graph.

    Implement this class to plug in different graph backends
    (JSON, Neo4j, SQLite, in-memory mock, etc.).
    """

    @abstractmethod
    def exists(self, node: str) -> bool:
        """Return True if ``node`` is present in the graph."""

    @abstractmethod
    def dependencies(self, node: str) -> frozenset[str]:
        """Return the set of declarations that ``node`` directly depends on.

        Corresponds to outgoing edges: ``node --USES--> dependency``.
        Returns an empty frozenset if ``node`` is not in the graph.
        """

    @abstractmethod
    def dependents(self, node: str) -> frozenset[str]:
        """Return the set of declarations that directly depend on ``node``.

        Corresponds to incoming edges: ``dependent --USES--> node``.
        Returns an empty frozenset if ``node`` is not in the graph.
        """

    @abstractmethod
    def expand(
        self,
        start: str,
        *,
        hops: int = 2,
        direction: str = "outgoing",
    ) -> dict[str, int]:
        """BFS expansion from ``start``.

        Parameters
        ----------
        start : str
            Starting declaration name.
        hops : int
            Maximum BFS depth.  0 returns only the start node.
        direction : str
            One of ``"outgoing"`` (follow USES edges), ``"incoming"``
            (follow reverse USES edges), or ``"both"``.

        Returns
        -------
        dict[str, int]
            Maps reachable node names to their minimum hop distance from
            ``start`` (inclusive: start node is at hop 0).
        """

    def get_related_lemmas(
        self,
        node: str,
        *,
        direction: str = "outgoing",
        hops: int = 2,
    ) -> dict[str, int]:
        """Convenience alias for :meth:`expand`.

        Returns all nodes reachable from ``node`` via dependency edges,
        keyed by their minimum hop distance.
        """
        return self.expand(node, hops=hops, direction=direction)

    def has_relation(self, source: str, relation: str, target: str) -> bool:
        """Return True if a direct edge of type ``relation`` exists.

        For the current schema only ``"USES"`` is defined.
        """
        if relation.upper() == "USES":
            return target in self.dependencies(source)
        return False

    @property
    @abstractmethod
    def node_count(self) -> int:
        """Number of declarations in the graph."""

    @property
    @abstractmethod
    def edge_count(self) -> int:
        """Number of directed edges in the graph."""


# ---------------------------------------------------------------------------
# JSON implementation
# ---------------------------------------------------------------------------

class JsonGraphProvider(GraphProvider):
    """Mathlib dependency graph loaded from a JSON file.

    This is the primary implementation for the current repository.
    It is designed to be replaceable by a Neo4j implementation later.

    Parameters
    ----------
    graph_path : str | Path
        Path to ``mathlib_dependencies.json``.
    """

    def __init__(self, graph_path: str | Path) -> None:
        self._graph_path = Path(graph_path)

        # Declaration set
        self._declarations: set[str] = set()

        # source → set(target)   (outgoing: source uses target)
        self._outgoing: dict[str, set[str]] = defaultdict(set)

        # target → set(source)   (incoming: source uses target)
        self._incoming: dict[str, set[str]] = defaultdict(set)

        self._edge_count = 0
        self._load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        data = json.loads(self._graph_path.read_text("utf-8"))
        self._declarations = set(data.get("declarations", []))
        removed_self_loops = 0
        for edge in data.get("edges", []):
            src = edge.get("source")
            tgt = edge.get("target")
            if not src or not tgt:
                continue
            if src == tgt:
                removed_self_loops += 1
                continue
            self._outgoing[src].add(tgt)
            self._incoming[tgt].add(src)
            self._edge_count += 1

    # ------------------------------------------------------------------
    # GraphProvider interface
    # ------------------------------------------------------------------

    def exists(self, node: str) -> bool:
        return (
            node in self._declarations
            or node in self._outgoing
            or node in self._incoming
        )

    def dependencies(self, node: str) -> frozenset[str]:
        return frozenset(self._outgoing.get(node, set()))

    def dependents(self, node: str) -> frozenset[str]:
        return frozenset(self._incoming.get(node, set()))

    def expand(
        self,
        start: str,
        *,
        hops: int = 2,
        direction: str = "outgoing",
    ) -> dict[str, int]:
        if direction not in {"outgoing", "incoming", "both"}:
            raise ValueError(
                f"direction must be 'outgoing', 'incoming', or 'both'; got {direction!r}"
            )
        if hops < 0:
            raise ValueError(f"hops must be >= 0, got {hops}")

        distances: dict[str, int] = {start: 0}
        queue: deque[str] = deque([start])

        while queue:
            node = queue.popleft()
            hop  = distances[node]
            if hop >= hops:
                continue

            neighbors: set[str] = set()
            if direction in {"outgoing", "both"}:
                neighbors.update(self._outgoing.get(node, set()))
            if direction in {"incoming", "both"}:
                neighbors.update(self._incoming.get(node, set()))

            for neighbor in neighbors:
                if neighbor not in distances:
                    distances[neighbor] = hop + 1
                    queue.append(neighbor)

        return distances

    @property
    def node_count(self) -> int:
        return len(self._declarations)

    @property
    def edge_count(self) -> int:
        return self._edge_count

    def __repr__(self) -> str:
        return (
            f"JsonGraphProvider("
            f"nodes={self.node_count}, "
            f"edges={self.edge_count})"
        )


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------

def load_graph(graph_path: str | Path) -> GraphProvider:
    """Load the Mathlib dependency graph.

    Currently returns a :class:`JsonGraphProvider`.

    Future: detect graph_path scheme (``neo4j://...``) and return the
    appropriate provider.
    """
    return JsonGraphProvider(graph_path)
