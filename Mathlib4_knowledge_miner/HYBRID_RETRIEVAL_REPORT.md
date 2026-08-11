# Mathlib4 Knowledge Miner — Hybrid Lemma Retrieval

## Overview

This system retrieves Mathlib 4 lemmas that are relevant to a proof state or
query embedding. It combines a fast approximate-nearest-neighbour semantic
stage with a logical/topological stage that exploits the Mathlib dependency
graph to rerank and expand candidates.

---

## Pipeline architecture

```
GNN lemma embeddings   (lemma_vectors.npy  — 59,555 × 512 float32)
          ↓
HNSW semantic retrieval   (lemma_hnsw.index, cosine space)
          ↓
lemma → Mathlib declaration mapping   (lemma_graph_mapping.json)
          ↓
Mathlib dependency graph expansion   (mathlib_dependencies.json)
          ↓
logical / topological scoring
          ↓
final ranked lemma recommendations
          ↓
evaluation against a semantic-only baseline
```

---

## Input artifacts

| File | Description |
|------|-------------|
| `lemma_vectors.npy` | GNN-generated embeddings, shape (59 555, 512), dtype float32. All vectors are unit-norm (norm = 1.0 for every row). |
| `lemma_ids.json` | Maps HNSW row index → lemma ID. |
| `lemma_hnsw.index` | hnswlib cosine-space index built from `lemma_vectors.npy`, M=32, ef_construction=200. |
| `lemmas.jsonl` | One JSON object per line: `{lemma_id, name, statement, …}`. |
| `mathlib_dependencies.json` | Mathlib dependency graph: 139 505 declarations, 157 763 valid directed edges (5 174 self-loops removed). |
| `lemma_graph_mapping.json` | Pre-computed lemma-ID → graph-declaration mapping. |

---

## GNN embeddings

Embeddings were produced by a graph neural network trained on the Mathlib
proof graph (contrastive training, 50 epochs).

The embedding matrix has:

- **59 555** lemmas
- **512** dimensions per lemma
- **unit L2 norm** (verified: all norms = 1.0, zero-vector count = 0)
- **21 718 unique rows** — many lemmas share identical embeddings due to
  clustering of topologically-equivalent graph roles

The embeddings are stored in `lemma_vectors.npy` and referenced by row index.

---

## HNSW semantic retrieval

**Class:** `HNSWRetriever` (`hnsw_retriever.py`)

The HNSW index uses **cosine distance** (`hnswlib`, space=`cosine`).
Distance reported by hnswlib ≈ `1 − cosine_similarity`.

Semantic similarity is recovered as:

```python
similarity = 1.0 − distance      # clipped to [0, 1]
```

### Diagnostic validation (TASK 1)

`diagnose_vectors.py` was run to validate:

| Check | Result |
|-------|--------|
| Shape | (59 555, 512) |
| dtype | float32 |
| Zero vectors | 0 |
| Unique rows | 21 718 |
| Norm min/mean/max | 1.000 / 1.000 / 1.000 |
| HNSW vs brute-force mean set overlap | **87.1 %** |
| Distance error (HNSW − BF) on common rows | 0.000000 |

**Verdict:** HNSW agrees well with brute-force cosine. The near-zero distances
for many lemma pairs are explained by the presence of 21 718 unique vectors
in a 59 555-entry matrix — many rows are duplicates. No modification to HNSW
is required.

---

## Lemma → graph mapping

**Class:** `LemmaGraphMapper` (`lemma_graph_mapper.py`)

Conservatively maps GNN lemma names to Mathlib declaration names using two
strategies only:

1. **Exact fully-qualified match** — name appears verbatim in the graph.
2. **Unique short-name match** — removing the namespace prefix yields exactly
   one candidate.

Ambiguous short names (multiple candidates) are **not guessed**.
Unmatched lemmas are reported separately.

| Category | Count |
|----------|-------|
| Exact matches | 10 153 |
| Unique short matches | 34 226 |
| Ambiguous | 4 840 |
| Unmatched | 10 336 |
| **Usable mappings** | **44 379** |

---

## Dependency graph

**Class:** `DependencyGraph` (`dependency_graph.py`)

In-memory directed graph loaded from `mathlib_dependencies.json`.

Edge semantics: `source --USES--> target`

| Statistic | Value |
|-----------|-------|
| Nodes (declarations) | 139 505 |
| Valid edges | 157 763 |
| Self-loops removed | 5 174 |
| Nodes with outgoing edges | 68 822 |
| Nodes with incoming edges | 32 075 |

Supported operations: `dependencies()`, `dependents()`, `expand()`,
`reachable()`, `neighborhood()`.

---

## Multi-hop expansion

**Method:** `LogicalTopologicalRetriever.expand_node()`

BFS expansion through outgoing dependency edges from a seed declaration.

Returns `{node: hop_distance}` for all reachable nodes within `max_hops`
(default 2).

Example: `Algebra.intTrace_eq_trace` expands to **16 nodes** within 2 hops.

---

## Logical / topological scoring

**Formula** (verified, immutable):

```
hop_score(hop)  = 1.0 / (hop + 1)      # hop 0 → 1.0, hop 1 → 0.5, hop 2 → 0.333…
```

For each semantic seed with score `s` and each discovered neighbour at hop `h ≥ 1`:

```
logical_contribution = s × hop_score(h)
```

The seed itself (hop = 0) is **not** added to `logical_score`; it is tracked
separately as `semantic_score`.

### Defect corrected

The expansion scoring loop (`for node, hop in neighborhood.items()`) was
inadvertently placed **outside** the `for candidate in semantic_candidates:`
loop. As a result only the last seed's neighborhood was scored; all earlier
seeds had `logical_score = 0.0`.

The fix was a single indentation correction (4 spaces) to bring the scoring
loop inside the candidate loop. No algorithmic change was made.

**Evidence:** Before the fix, every candidate showed `logical_score = 0.0000`
in the final output. After the fix, candidates discovered via graph expansion
receive correct non-zero logical scores, and graph-discovered candidates appear
in the final top-K.

---

## Final score

```
final_score = (
    1.00 × semantic_score
  + 0.75 × logical_score
  + 0.50 × frequency_score
  + 0.25 × topological_score
)
```

Where:
- `semantic_score` = HNSW similarity (best across seeds that map to this node)
- `logical_score` = accumulated `seed_semantic × hop_score(hop)` over all seeds
- `frequency_score` = times this node appeared (as seed or neighbour) / max frequency
- `topological_score` = `hop_score(best_hop)`

These weights are fixed unless an evaluation experiment demonstrates a better
setting.

---

## Evaluation methodology

**Script:** `evaluation/lemma_retrieval_bench.py` and `evaluation/run_benchmark.sh`

**Important Note:** The official mentor baseline of 797 validation targets has **not yet been reproduced**. The mentor's 797-example PyG validation dataset (generated via the `atp-improvements` branch) is missing from the provided artifacts. 

Instead, the benchmark was executed on a locally generated 1,163-target validation set. The current retrieval results are **not directly comparable** to the mentor's official baseline. The framework is fully parameterized to run the official PyG test set once it is supplied.

### Aggregate results (Local 1,163-Target Dataset)

The semantic-only baseline produces the following approximate recall on the 1,163-target set:
- R@200: ~0.24 
- R@500: ~0.35 
- R@1000: ~0.44 
- R@2000: ~0.54 
- R@5000: ~0.70 

The hybrid reranker operates over this baseline to adjust rankings using multi-hop graph expansion, systematically boosting mapped candidates from the dependency graph and surfacing logically supported neighbors.

---

## Unit tests

**File:** `test_hybrid_retriever.py`

35 tests covering:

| Group | Tests |
|-------|-------|
| `_hop_score` values | 5 |
| Logical score accumulation | 6 |
| `DependencyGraph.expand` | 11 |
| Semantic score conversion | 5 |
| Final score formula | 2 |

All 35 tests pass.

---

## Current limitations

1. **No labeled ground truth.** Precision/recall cannot be computed without
   human annotations of relevant lemmas per query. The structural metrics
   (overlap, rank change, graph support) are a proxy.

2. **Duplicate embeddings.** 59 555 lemmas share only 21 718 unique vectors.
   Many lemmas are near-identical in embedding space regardless of their
   mathematical content. This is a property of the GNN training, not of HNSW.

3. **Mapping coverage 66 %.** ~10 336 lemmas have no match in the dependency
   graph (unmatched), and 4 840 have ambiguous short names. These receive no
   logical score boost.

4. **Weights are untuned.** The final score weights (1.0 / 0.75 / 0.50 / 0.25)
   are reasonable defaults. A grid search over labeled queries could improve
   them.

5. **No incoming-edge expansion.** The current pipeline follows only outgoing
   (dependency) edges. Expanding in the incoming direction (dependents) would
   surface lemmas that _use_ the semantic candidates, which may also be
   relevant.

---

## Scripts reference

| Script | Purpose |
|--------|---------|
| `logical_topological_retriever.py` | Main hybrid retrieval pipeline |
| `hnsw_retriever.py` | HNSW semantic stage |
| `lemma_graph_mapper.py` | Lemma → graph declaration mapping |
| `dependency_graph.py` | Mathlib dependency graph |
| `diagnose_vectors.py` | Vector/HNSW diagnostic (TASK 1) |
| `evaluate_hybrid_retrieval.py` | Structural evaluation (TASK 2) |
| `test_hybrid_retriever.py` | Unit tests (TASK 3) |
