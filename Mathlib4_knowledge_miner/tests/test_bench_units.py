"""
tests/test_bench_units.py
=========================

Unit tests for the lemma retrieval benchmark.

Tests cover:
    - Argument classification (gold label extraction)
    - Local-hypothesis-only filtering
    - Lemma ID mapping
    - MRR calculation
    - Recall@k calculation
    - Semantic-only retrieval (mock)
    - Hybrid reranking logic
    - Handling missing/unresolved lemma IDs
    - Median computation
    - Edge cases (empty args, no library lemmas, perfect recall)

Run with:
    python -m pytest tests/test_bench_units.py -v

No GNN/torch_geometric/datasets required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import numpy as np

# Add repo root to path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.gold_labels import (
    ArgCategory,
    ClassifiedArg,
    TacticGoldLabel,
    classify_argument,
    extract_gold_labels,
    parse_tactic_arguments,
    _extract_hypothesis_names,
)
from evaluation.metrics import (
    MetricSummary,
    RetrievalResult,
    _median,
    compute_global_mrr,
    compute_metrics,
    find_first_hit_rank,
)


# ============================================================
# Argument parsing tests
# ============================================================

class TestParseArguments:
    def test_apply_lemma(self):
        tactic, args = parse_tactic_arguments("apply Nat.add_comm")
        assert tactic == "apply"
        assert args == ["Nat.add_comm"]

    def test_rw_with_brackets(self):
        tactic, args = parse_tactic_arguments("rw [foo, bar]")
        assert tactic == "rw"
        assert "foo" in args
        assert "bar" in args

    def test_simp_no_args(self):
        tactic, args = parse_tactic_arguments("simp")
        assert tactic == "simp"
        assert args == []

    def test_simp_only(self):
        tactic, args = parse_tactic_arguments("simp only [h1, h2]")
        assert tactic == "simp"
        assert "h1" in args
        assert "h2" in args

    def test_exact_hypothesis(self):
        tactic, args = parse_tactic_arguments("exact h")
        assert tactic == "exact"
        assert args == ["h"]

    def test_empty_tactic(self):
        tactic, args = parse_tactic_arguments("")
        assert tactic == "<EMPTY_TACTIC>"
        assert args == []

    def test_have_two_args(self):
        tactic, args = parse_tactic_arguments("have h : Nat := foo")
        assert tactic == "have"
        assert "h" in args

    def test_linarith_no_args(self):
        tactic, args = parse_tactic_arguments("linarith")
        assert tactic == "linarith"
        assert args == []


# ============================================================
# Hypothesis name extraction tests
# ============================================================

class TestHypothesisNames:
    def test_unicode_turnstile(self):
        state = "h : Nat\nh2 : h > 0\n⊢ h + 1 > 0"
        names = _extract_hypothesis_names(state)
        assert "h" in names
        assert "h2" in names

    def test_ascii_turnstile(self):
        state = "n : Nat |- n ≥ 0"
        names = _extract_hypothesis_names(state)
        assert "n" in names

    def test_no_hypotheses(self):
        state = "⊢ 1 + 1 = 2"
        names = _extract_hypothesis_names(state)
        assert names == set()

    def test_multiple_hypotheses(self):
        state = "a : ℕ\nb : a > 0\nc : b → False\n⊢ False"
        names = _extract_hypothesis_names(state)
        assert "a" in names
        assert "b" in names
        assert "c" in names


# ============================================================
# Argument classification tests
# ============================================================

class TestClassifyArgument:
    LEMMA_INDEX = {"Nat.add_comm": 42, "Finset.sum_range": 100}
    HYP_NAMES = {"h", "h1", "n", "hn"}

    def test_library_lemma(self):
        result = classify_argument(
            "Nat.add_comm",
            hypothesis_names=self.HYP_NAMES,
            lemma_name_index=self.LEMMA_INDEX,
        )
        assert result.category == ArgCategory.LIBRARY_LEMMA
        assert result.lemma_id == 42

    def test_local_hypothesis(self):
        result = classify_argument(
            "h",
            hypothesis_names=self.HYP_NAMES,
            lemma_name_index=self.LEMMA_INDEX,
        )
        assert result.category == ArgCategory.LOCAL_HYPOTHESIS
        assert result.lemma_id == -1

    def test_unresolved(self):
        result = classify_argument(
            "someUnknownThing",
            hypothesis_names=self.HYP_NAMES,
            lemma_name_index=self.LEMMA_INDEX,
        )
        assert result.category == ArgCategory.UNRESOLVED
        assert result.lemma_id == -1

    def test_raw_expression_numeric(self):
        result = classify_argument(
            "42",
            hypothesis_names=self.HYP_NAMES,
            lemma_name_index=self.LEMMA_INDEX,
        )
        assert result.category == ArgCategory.RAW_EXPRESSION

    def test_local_hyp_takes_priority_over_unresolved(self):
        # If a name is BOTH in hyps AND lemma_index (unlikely, but tested):
        # local hypothesis wins (classified first)
        result = classify_argument(
            "h",
            hypothesis_names={"h"},
            lemma_name_index={"h": 999},  # h also appears in corpus
        )
        assert result.category == ArgCategory.LOCAL_HYPOTHESIS


# ============================================================
# Gold label extraction tests
# ============================================================

class TestExtractGoldLabels:
    LEMMA_INDEX = {"Nat.add_comm": 10, "List.length_map": 20}

    def test_apply_library_lemma(self):
        gold = extract_gold_labels(
            "apply Nat.add_comm",
            "n : Nat\n⊢ n + 0 = 0 + n",
            lemma_name_index=self.LEMMA_INDEX,
        )
        assert gold.has_library_lemma
        assert 10 in gold.library_lemma_ids

    def test_exact_local_hypothesis(self):
        gold = extract_gold_labels(
            "exact h",
            "h : Nat\n⊢ Nat",
            lemma_name_index=self.LEMMA_INDEX,
        )
        assert not gold.has_library_lemma
        assert gold.is_local_only

    def test_simp_no_args(self):
        gold = extract_gold_labels(
            "simp",
            "⊢ 1 + 1 = 2",
            lemma_name_index=self.LEMMA_INDEX,
        )
        # No args → not local, not library, just empty
        assert not gold.has_library_lemma
        assert not gold.is_local_only

    def test_rw_with_lemma(self):
        gold = extract_gold_labels(
            "rw [List.length_map]",
            "l : List Nat\n⊢ (l.map id).length = l.length",
            lemma_name_index=self.LEMMA_INDEX,
        )
        assert gold.has_library_lemma
        assert 20 in gold.library_lemma_ids

    def test_multiple_args_mixed(self):
        """One local hyp + one library lemma → counts as library_lemma."""
        gold = extract_gold_labels(
            "rw [h, Nat.add_comm]",
            "h : x = y\n⊢ y + 0 = 0 + y",
            lemma_name_index=self.LEMMA_INDEX,
        )
        assert gold.has_library_lemma
        assert 10 in gold.library_lemma_ids

    def test_local_only_not_counted(self):
        """Proof state with only local hypothesis args should be excluded."""
        gold = extract_gold_labels(
            "apply h1",
            "h1 : P → Q\n⊢ Q",
            lemma_name_index=self.LEMMA_INDEX,
        )
        assert not gold.has_library_lemma
        assert gold.is_local_only


# ============================================================
# Metrics tests
# ============================================================

class TestFindFirstHitRank:
    def test_hit_at_rank_1(self):
        assert find_first_hit_rank([5], [5, 3, 1]) == 1

    def test_hit_at_rank_3(self):
        assert find_first_hit_rank([1], [5, 3, 1]) == 3

    def test_no_hit(self):
        assert find_first_hit_rank([99], [5, 3, 1]) is None

    def test_multiple_gold_ids_first_wins(self):
        # gold = [3, 1]; rank of 3 is 2, rank of 1 is 3 → returns 2
        assert find_first_hit_rank([3, 1], [5, 3, 1]) == 2

    def test_empty_retrieved(self):
        assert find_first_hit_rank([1], []) is None

    def test_empty_gold(self):
        assert find_first_hit_rank([], [1, 2, 3]) is None


class TestMedian:
    def test_odd(self):
        assert _median([1, 3, 5]) == 3.0

    def test_even(self):
        assert _median([1, 2, 3, 4]) == 2.5

    def test_single(self):
        assert _median([7]) == 7.0

    def test_empty(self):
        import math
        assert math.isnan(_median([]))


class TestComputeMetrics:
    def _make_results(self, hit_ranks: list[int | None]) -> list[RetrievalResult]:
        """Make a list of non-excluded results with given hit ranks."""
        return [
            RetrievalResult(
                query_index=i,
                gold_lemma_ids=[i],
                retrieved_lemma_ids=list(range(6000)),
                hit_rank=hr,
            )
            for i, hr in enumerate(hit_ranks)
        ]

    def test_perfect_recall(self):
        results = self._make_results([1, 2, 3, 4, 5])
        summary = compute_metrics(results, k_values=[5, 10])
        assert summary.recall[5] == 1.0
        assert summary.recall[10] == 1.0
        assert summary.target_samples == 5

    def test_zero_recall(self):
        results = self._make_results([None, None, None])
        summary = compute_metrics(results, k_values=[200])
        assert summary.recall[200] == 0.0
        assert summary.mrr[200] == 0.0

    def test_partial_recall_at_k(self):
        # 2 out of 4 hit within k=2, 3 out of 4 hit within k=5
        results = self._make_results([1, 2, 5, None])
        summary = compute_metrics(results, k_values=[2, 5])
        assert summary.recall[2] == 2 / 4
        assert summary.recall[5] == 3 / 4

    def test_mrr_simple(self):
        # hits at rank 1, 2, and None → MRR = (1/1 + 1/2 + 0) / 3
        results = self._make_results([1, 2, None])
        summary = compute_metrics(results, k_values=[5000])
        expected_mrr = (1.0 + 0.5) / 3
        assert abs(summary.mrr[5000] - expected_mrr) < 1e-9

    def test_exclusion_counting(self):
        excluded = [
            RetrievalResult(
                query_index=0,
                gold_lemma_ids=[],
                retrieved_lemma_ids=[],
                hit_rank=None,
                excluded=True,
                exclusion_reason="local_hypothesis_only",
            ),
            RetrievalResult(
                query_index=1,
                gold_lemma_ids=[],
                retrieved_lemma_ids=[],
                hit_rank=None,
                excluded=True,
                exclusion_reason="unresolved_only",
            ),
        ]
        valid = self._make_results([1])
        summary = compute_metrics(excluded + valid, k_values=[200])
        assert summary.excluded_local_hyp == 1
        assert summary.excluded_unresolved == 1
        assert summary.target_samples == 1

    def test_hit_counts_at_k(self):
        results = self._make_results([100, 250, 600, None])
        summary = compute_metrics(results, k_values=[200, 500])
        assert summary.hit_counts[200] == 1   # only rank 100 <= 200
        assert summary.hit_counts[500] == 2   # ranks 100, 250 <= 500

    def test_median_hit_rank(self):
        results = self._make_results([10, 20, 30])
        summary = compute_metrics(results, k_values=[100])
        assert summary.median_hit_rank[100] == 20.0

    def test_mean_hit_rank(self):
        results = self._make_results([10, 20, 30])
        summary = compute_metrics(results, k_values=[100])
        assert abs(summary.mean_hit_rank[100] - 20.0) < 1e-9

    def test_no_results(self):
        summary = compute_metrics([], k_values=[200])
        assert summary.target_samples == 0
        assert summary.recall[200] == 0.0


class TestGlobalMRR:
    def test_global_mrr(self):
        results = [
            RetrievalResult(0, [1], [], hit_rank=1),
            RetrievalResult(1, [2], [], hit_rank=2),
            RetrievalResult(2, [3], [], hit_rank=None),
        ]
        # MRR = (1 + 0.5 + 0) / 3
        expected = (1.0 + 0.5) / 3
        assert abs(compute_global_mrr(results) - expected) < 1e-9

    def test_global_mrr_all_excluded(self):
        results = [
            RetrievalResult(0, [], [], None, excluded=True, exclusion_reason="local_hypothesis_only"),
        ]
        assert compute_global_mrr(results) == 0.0


# ============================================================
# Semantic retrieval mock tests
# ============================================================

class TestFAISSRetrieverMock:
    """Test FAISSRetriever interface with a mock index."""

    def test_search_returns_lemma_ids(self):
        """Mock test: given an index that returns row indices, check ID mapping."""
        # Simulate what FAISSRetriever does internally
        lemma_ids = [100, 200, 300, 400, 500]

        class MockIndex:
            d = 4
            def search(self, query, k):
                scores = np.array([[0.9, 0.8, 0.7, 0.6, 0.5]])
                indices = np.array([[0, 1, 2, 3, 4]])
                return scores, indices

        class MockRetriever:
            def __init__(self):
                self.index = MockIndex()
                self.lemma_ids = lemma_ids
                self._num_lemmas = len(lemma_ids)
                self.normalize_queries = False

            def search(self, query_vector, *, k=5):
                query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
                _, indices = self.index.search(query, min(k, self._num_lemmas))
                return [self.lemma_ids[int(idx)] for idx in indices[0] if 0 <= int(idx) < self._num_lemmas]

        retriever = MockRetriever()
        result = retriever.search(np.zeros(4), k=5)
        assert result == [100, 200, 300, 400, 500]

    def test_k_clamped_to_num_lemmas(self):
        """Requesting k > num_lemmas should not error."""
        lemma_ids = [1, 2, 3]

        class MockIndex:
            d = 4
            def search(self, query, k):
                return np.array([[0.9, 0.8, 0.7]]), np.array([[0, 1, 2]])

        class MockRetriever:
            def __init__(self):
                self.index = MockIndex()
                self.lemma_ids = lemma_ids
                self._num_lemmas = len(lemma_ids)
                self.normalize_queries = False

            def search(self, query_vector, *, k=1000):
                k_c = min(k, self._num_lemmas)
                _, indices = self.index.search(None, k_c)
                return [self.lemma_ids[int(i)] for i in indices[0] if 0 <= int(i) < self._num_lemmas]

        retriever = MockRetriever()
        result = retriever.search(np.zeros(4), k=10000)
        assert len(result) == 3


# ============================================================
# Hybrid reranking logic tests
# ============================================================

class TestHybridReranking:
    """Test the hybrid scoring arithmetic."""

    def test_semantic_only_candidate_scores_higher(self):
        """A direct semantic hit (best_hop=0) should outscore a pure graph expansion."""
        from evaluation.hybrid_reranker import _hop_score

        # Direct hit
        sem_score = 0.9
        direct_score = 1.0 * sem_score + 0.75 * 0.0 + 0.50 * 1.0 + 0.25 * _hop_score(0)
        # Graph expanded at hop=1, sem_score from seed=0.8
        expansion_score = (
            0.75 * (0.8 * _hop_score(1))   # logical
            + 0.50 * 0.5                    # freq
            + 0.25 * _hop_score(1)          # hop
        )
        assert direct_score > expansion_score

    def test_hop_score_decreases_with_hop(self):
        from evaluation.hybrid_reranker import _hop_score
        assert _hop_score(0) == 1.0
        assert _hop_score(1) < _hop_score(0)
        assert _hop_score(2) < _hop_score(1)

    def test_bfs_expand_max_hops(self):
        """BFS should not expand beyond max_hops."""
        from evaluation.hybrid_reranker import _bfs_expand

        # A → B → C → D (chain)
        outgoing = {
            "A": {"B"},
            "B": {"C"},
            "C": {"D"},
        }
        result = _bfs_expand("A", outgoing, max_hops=2)
        assert "A" in result and result["A"] == 0
        assert "B" in result and result["B"] == 1
        assert "C" in result and result["C"] == 2
        assert "D" not in result  # beyond max_hops=2

    def test_bfs_expand_empty_graph(self):
        from evaluation.hybrid_reranker import _bfs_expand
        result = _bfs_expand("X", {}, max_hops=2)
        assert result == {"X": 0}

    def test_bfs_expand_cycle_handled(self):
        """BFS should not loop on cycles."""
        from evaluation.hybrid_reranker import _bfs_expand
        outgoing = {"A": {"B"}, "B": {"A"}}
        result = _bfs_expand("A", outgoing, max_hops=3)
        assert "A" in result
        assert "B" in result
        assert result["A"] == 0
        assert result["B"] == 1

    def test_mapping_load(self):
        """Test mapping file loading with the reference format."""
        import json
        import tempfile
        import os

        from evaluation.hybrid_reranker import _load_mapping

        mapping = {"mapping": {"10": "Mathlib.Foo", "20": "Mathlib.Bar"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(mapping, f)
            tmp_path = f.name

        try:
            l2g, g2l = _load_mapping(tmp_path)
            assert l2g[10] == "Mathlib.Foo"
            assert g2l["Mathlib.Foo"] == 10
            assert l2g[20] == "Mathlib.Bar"
        finally:
            os.unlink(tmp_path)


# ============================================================
# Lemma ID mapping tests
# ============================================================

class TestLemmaIDMapping:
    def test_build_index_first_occurrence_wins(self):
        """If a name appears twice, the first lemma_id wins."""
        corpus = {10: "Nat.add_comm", 20: "Nat.add_comm", 30: "List.length_map"}
        name_index = {}
        for lid, name in corpus.items():
            if name not in name_index:
                name_index[name] = lid
        assert name_index["Nat.add_comm"] == 10

    def test_missing_name_returns_minus_one(self):
        lemma_name_index = {"Nat.add_comm": 42}
        result = lemma_name_index.get("UnknownLemma", -1)
        assert result == -1


# ============================================================
# Handling missing/unresolved IDs
# ============================================================

class TestMissingUnresolved:
    def test_unresolved_is_excluded(self):
        """A tactic with only unresolved args should not contribute to denominator."""
        gold = extract_gold_labels(
            "apply completelyfake",
            "⊢ True",
            lemma_name_index={},
        )
        assert not gold.has_library_lemma

    def test_empty_tactic_excluded(self):
        gold = extract_gold_labels(
            "",
            "⊢ True",
            lemma_name_index={},
        )
        assert not gold.has_library_lemma

    def test_no_corpus_entry_unresolved(self):
        gold = extract_gold_labels(
            "exact SomeLemmaNotInCorpus",
            "⊢ P",
            lemma_name_index={"OtherLemma": 1},
        )
        assert not gold.has_library_lemma

    def test_retrieved_with_invalid_index_skipped(self):
        """If FAISS returns -1 (invalid), it should not match any gold."""
        result = find_first_hit_rank([5], [-1, -1, 5])
        # -1 should not be in gold_set (gold = {5}), and 5 appears at rank 3
        assert result == 3


# ============================================================
# Multiple gold lemma IDs per tactic
# ============================================================

class TestMultipleGoldLemmas:
    LEMMA_INDEX = {"Nat.add_comm": 10, "List.length_map": 20, "Finset.sum": 30}

    def test_rw_two_library_lemmas(self):
        """rw [A, B] where both A and B are library lemmas → both in gold."""
        gold = extract_gold_labels(
            "rw [Nat.add_comm, List.length_map]",
            "l : List Nat\n⊢ l.length + 0 = 0 + l.length",
            lemma_name_index=self.LEMMA_INDEX,
        )
        assert gold.has_library_lemma
        assert 10 in gold.library_lemma_ids
        assert 20 in gold.library_lemma_ids
        assert len(gold.library_lemma_ids) == 2

    def test_hit_rank_uses_first_hit_among_multiple_gold(self):
        """If gold = [10, 20] and retrieved = [100, 20, 10], hit rank = 2."""
        rank = find_first_hit_rank([10, 20], [100, 20, 10])
        assert rank == 2  # 20 appears first at position 2

    def test_hit_rank_when_only_second_gold_hit(self):
        """If gold = [10, 20] and only 20 appears in top-k, hit rank = that rank."""
        rank = find_first_hit_rank([10, 20], [100, 200, 20, 999])
        assert rank == 3

    def test_both_gold_not_in_retrieved(self):
        """If neither gold lemma appears, hit_rank = None."""
        rank = find_first_hit_rank([10, 20], [100, 200, 300])
        assert rank is None

    def test_multiple_gold_recall_at_k(self):
        """Recall@k should be 1 if ANY gold lemma is in top-k."""
        results = [
            RetrievalResult(0, [10, 20], [], hit_rank=3),   # gold 20 at rank 3 ≤ 5
            RetrievalResult(1, [30], [],    hit_rank=None),  # miss
        ]
        summary = compute_metrics(results, k_values=[5])
        # 1 out of 2 hit within k=5
        assert summary.recall[5] == 0.5

    def test_multiple_gold_deduplicated(self):
        """If tactic has the same library lemma twice, ID should appear once."""
        gold = extract_gold_labels(
            "rw [Nat.add_comm, Nat.add_comm]",
            "⊢ 1 + 2 = 2 + 1",
            lemma_name_index=self.LEMMA_INDEX,
        )
        assert gold.library_lemma_ids.count(10) == 1


# ============================================================
# Missing graph mapping — hybrid must not remove gold lemma
# ============================================================

class TestHybridMissingGraphMapping:
    """Hybrid reranker must handle missing graph entries gracefully.

    If a gold lemma has no graph entry, it must remain in the candidate list
    with its original semantic score — it should never be suppressed.
    """

    def _make_retriever(self, *, include_gold_in_graph=False):
        """Build a HybridReranker with a tiny mock FAISS + mapping."""
        import json, tempfile, os
        from evaluation.hybrid_reranker import HybridReranker
        from evaluation.faiss_retriever import FAISSRetriever
        import faiss

        # Tiny 4-dim index with 3 lemmas
        dim = 4
        vecs = np.array([
            [1., 0., 0., 0.],   # lemma_id=10 (gold)
            [0., 1., 0., 0.],   # lemma_id=20
            [0., 0., 1., 0.],   # lemma_id=30
        ], dtype=np.float32)
        # Normalize
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

        idx = faiss.IndexFlatIP(dim)
        idx.add(vecs)

        lemma_ids = [10, 20, 30]

        # Build mapping: only 20 and 30 are in graph; 10 (gold) is MISSING
        if include_gold_in_graph:
            mapping = {"mapping": {"10": "Mathlib.Gold", "20": "Mathlib.B", "30": "Mathlib.C"}}
            outgoing = {"Mathlib.B": {"Mathlib.C"}}
        else:
            mapping = {"mapping": {"20": "Mathlib.B", "30": "Mathlib.C"}}
            outgoing = {"Mathlib.B": {"Mathlib.C"}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as mf:
            json.dump(mapping, mf)
            map_path = mf.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as gf:
            json.dump({"edges": [{"source": k, "target": v}
                                  for k, vs in outgoing.items() for v in vs]}, gf)
            graph_path = gf.name

        try:
            class MockFAISS:
                def __init__(self):
                    self.index = idx
                    self.lemma_ids = lemma_ids
                    self._num_lemmas = 3
                    self.normalize_queries = True

                def search(self, query_vector, *, k=5):
                    q = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
                    _, inds = self.index.search(q, min(k, self._num_lemmas))
                    return [self.lemma_ids[int(i)] for i in inds[0] if 0 <= int(i) < self._num_lemmas]

            from evaluation.hybrid_reranker import _load_mapping, _load_graph, HybridReranker
            l2g, g2l = _load_mapping(map_path)
            og = _load_graph(graph_path)
            reranker = HybridReranker(MockFAISS(), l2g, g2l, og)
            return reranker, map_path, graph_path
        except Exception:
            os.unlink(map_path)
            os.unlink(graph_path)
            raise

    def test_gold_not_in_graph_still_retrieved(self):
        """Gold lemma (id=10) has no graph entry. Hybrid must still return it."""
        import os
        reranker, mp, gp = self._make_retriever(include_gold_in_graph=False)
        try:
            # Query vector closest to lemma_id=10 (gold)
            query = np.array([1., 0., 0., 0.], dtype=np.float32)
            results = reranker.hybrid_retrieve(query, semantic_k=3, final_k=3)
            # Gold lemma (id=10) must appear in results
            assert 10 in results, f"Gold lemma 10 missing from hybrid results: {results}"
        finally:
            os.unlink(mp)
            os.unlink(gp)

    def test_missing_graph_entry_no_exception(self):
        """Hybrid must not raise when most lemmas lack graph mappings."""
        import os
        reranker, mp, gp = self._make_retriever(include_gold_in_graph=False)
        try:
            query = np.array([0., 1., 0., 0.], dtype=np.float32)
            # Should not raise
            results = reranker.hybrid_retrieve(query, semantic_k=3, final_k=3)
            assert isinstance(results, list)
        finally:
            os.unlink(mp)
            os.unlink(gp)


# ============================================================
# FAISS ranking correctness
# ============================================================

class TestFAISSRanking:
    """Verify that FAISS returns results in descending similarity order."""

    def test_faiss_ranks_in_similarity_order(self):
        """Nearest neighbor gets rank 1, second-nearest rank 2, etc."""
        try:
            import faiss
        except ImportError:
            import pytest
            pytest.skip("faiss not installed")

        dim = 4
        vecs = np.array([
            [0.9, 0.1, 0.0, 0.0],  # id=100, closest to query
            [0.5, 0.5, 0.0, 0.0],  # id=200
            [0.1, 0.9, 0.0, 0.0],  # id=300, farthest
        ], dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

        idx = faiss.IndexFlatIP(dim)
        idx.add(vecs)

        query = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        scores, indices = idx.search(query, 3)

        assert indices[0][0] == 0  # lemma id=100 vector at row 0 is closest
        assert scores[0][0] > scores[0][1] > scores[0][2]

    def test_faiss_search_via_retriever(self):
        """FAISSRetriever.search returns lemma_ids in score order."""
        try:
            import faiss
        except ImportError:
            import pytest
            pytest.skip("faiss not installed")

        import json, tempfile, os, numpy as np
        from evaluation.faiss_retriever import FAISSRetriever

        dim = 4
        vecs = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=np.float32)

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            idx = faiss.IndexFlatIP(dim)
            idx.add(vecs)
            faiss.write_index(idx, str(td / "faiss.index"))
            (td / "lemma_ids.json").write_text(json.dumps([10, 20, 30]))
            np.save(str(td / "lemma_vectors.npy"), vecs)
            (td / "manifest.json").write_text(json.dumps({"normalize": False}))

            retriever = FAISSRetriever.load(td)
            # Query closest to lemma 20 (row 1)
            query = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
            result = retriever.search(query, k=3)
            assert result[0] == 20


# ============================================================
# 797-target denominator reproducibility
# ============================================================

class TestDenominator797:
    """
    Verify that the gold-label logic produces the correct denominator
    for a representative sample of val proof states.

    We cannot run the full val split here (needs datasets/GNN), but we can
    verify that:
    1. simp/ring/tauto/linarith with no args are excluded
    2. exact h (local hyp) is excluded
    3. apply Lib.lemma is included
    4. rw [h, Lib.lemma] is included (has ≥1 library lemma)
    5. omega with no args is excluded
    6. constructor with no args is excluded
    """
    LEMMA_INDEX = {"Nat.add_zero": 5, "List.cons_append": 7, "Finset.card_eq_zero": 9}

    def _check(self, tactic: str, state: str, expect_included: bool, label: str):
        gold = extract_gold_labels(tactic, state, lemma_name_index=self.LEMMA_INDEX)
        if expect_included:
            assert gold.has_library_lemma, f"Expected included but excluded: {label}"
        else:
            assert not gold.has_library_lemma, f"Expected excluded but included: {label}"

    def test_simp_excluded(self):
        self._check("simp", "⊢ 1 + 1 = 2", False, "simp")

    def test_ring_excluded(self):
        self._check("ring", "⊢ (a+b)^2 = a^2+2*a*b+b^2", False, "ring")

    def test_linarith_excluded(self):
        self._check("linarith", "h : n > 0\n⊢ n ≥ 1", False, "linarith")

    def test_omega_excluded(self):
        self._check("omega", "⊢ 2 + 2 = 4", False, "omega")

    def test_exact_local_hyp_excluded(self):
        self._check("exact h", "h : P\n⊢ P", False, "exact h")

    def test_apply_library_included(self):
        self._check("apply Nat.add_zero", "n : Nat\n⊢ n + 0 = n",
                    True, "apply Nat.add_zero")

    def test_rw_mix_included(self):
        self._check("rw [h, Nat.add_zero]", "h : x = y\n⊢ y + 0 = x",
                    True, "rw [h, Nat.add_zero]")

    def test_constructor_excluded(self):
        self._check("constructor", "⊢ True ∧ True", False, "constructor")

    def test_intro_local_excluded(self):
        self._check("intro h", "⊢ P → P", False, "intro h")

    def test_exact_lib_included(self):
        self._check("exact List.cons_append", "⊢ (a :: l1).append l2 = a :: l1.append l2",
                    True, "exact List.cons_append")

    def test_denominator_count_in_sample(self):
        """Out of 10 sample tactics, count how many are included."""
        samples = [
            ("simp", "⊢ True", False),
            ("apply Nat.add_zero", "⊢ n + 0 = n", True),
            ("exact h", "h : P\n⊢ P", False),
            ("rw [List.cons_append]", "⊢ (a::l).append l2 = a::l.append l2", True),
            ("linarith", "h : n > 0\n⊢ n ≥ 1", False),
            ("exact Finset.card_eq_zero", "⊢ s.card = 0 ↔ s = ∅", True),
            ("omega", "⊢ 0 + n = n", False),
            ("rw [h1, h2]", "h1 : a=b\nh2 : b=c\n⊢ a = c", False),
            ("apply Nat.add_zero", "m : Nat\n⊢ m + 0 = m", True),
            ("constructor", "⊢ True ∧ True", False),
        ]
        included = sum(
            1 for tactic, state, expected in samples
            if extract_gold_labels(tactic, state, lemma_name_index=self.LEMMA_INDEX).has_library_lemma
        )
        expected_included = sum(1 for _, _, e in samples if e)
        assert included == expected_included, (
            f"Denominator mismatch: got {included}, expected {expected_included}"
        )

# ============================================================
# Anchor tag gold label tests (LeanDojo real format)
# ============================================================

class TestAnchorTagGoldLabels:
    """Verify that <a>lemmaName</a> tags are the sole gold label source."""

    LEMMA_INDEX = {
        "Nat.add_comm": 10,
        "List.length_map": 20,
        "WithLp.prod_norm_eq_add": 30,
        "Polynomial.eraseLead_support": 40,
        "Finset.mem_erase": 50,
    }

    def test_rw_with_anchor_tags(self):
        gold = extract_gold_labels(
            "rw [<a>Nat.add_comm</a>, h]",
            "h : x = y\n⊢ y + 0 = x",
            lemma_name_index=self.LEMMA_INDEX,
        )
        assert gold.has_library_lemma
        assert gold.library_lemma_ids == [10]

    def test_simp_anchor_with_extra_arg(self):
        gold = extract_gold_labels(
            "simp [<a>WithLp.prod_norm_eq_add</a> (p.symm)]",
            "⊢ True",
            lemma_name_index=self.LEMMA_INDEX,
        )
        assert gold.has_library_lemma
        assert 30 in gold.library_lemma_ids

    def test_two_anchors_both_in_corpus(self):
        gold = extract_gold_labels(
            "rw [<a>Polynomial.eraseLead_support</a>, <a>Finset.mem_erase</a>] at h",
            "⊢ True",
            lemma_name_index=self.LEMMA_INDEX,
        )
        assert gold.has_library_lemma
        assert 40 in gold.library_lemma_ids
        assert 50 in gold.library_lemma_ids

    def test_anchor_not_in_corpus_excluded(self):
        gold = extract_gold_labels(
            "apply <a>UnknownLemma</a>",
            "⊢ True",
            lemma_name_index=self.LEMMA_INDEX,
        )
        assert not gold.has_library_lemma

    def test_no_anchor_tags_local_hyp_not_counted(self):
        gold = extract_gold_labels(
            "exact h",
            "h : P\n⊢ P",
            lemma_name_index=self.LEMMA_INDEX,
        )
        assert not gold.has_library_lemma

    def test_anchor_duplicate_deduplicated(self):
        gold = extract_gold_labels(
            "rw [<a>Nat.add_comm</a>, <a>Nat.add_comm</a>]",
            "⊢ True",
            lemma_name_index=self.LEMMA_INDEX,
        )
        assert gold.library_lemma_ids.count(10) == 1

    def test_simpa_using_local_hyp_gold_from_anchor_only(self):
        gold = extract_gold_labels(
            "simpa only [<a>Nat.add_comm</a>] using hfg",
            "hfg : P\n⊢ Q",
            lemma_name_index=self.LEMMA_INDEX,
        )
        assert gold.has_library_lemma
        assert 10 in gold.library_lemma_ids
        assert len(gold.library_lemma_ids) == 1


class TestAnchorTagDenominator:
    LEMMA_INDEX = {"Nat.add_zero": 5, "List.cons_append": 7}

    def test_anchor_tactics_counted_correctly(self):
        tactics_states = [
            ("apply <a>Nat.add_zero</a>", "⊢ n + 0 = n", True),
            ("simp", "⊢ True", False),
            ("exact h", "h : P\n⊢ P", False),
            ("rw [<a>List.cons_append</a>]", "⊢ True", True),
            ("omega", "⊢ 0 = 0", False),
            ("rw [<a>NotInCorpus</a>]", "⊢ True", False),
        ]
        included = sum(
            1 for tactic, state, _ in tactics_states
            if extract_gold_labels(tactic, state, lemma_name_index=self.LEMMA_INDEX).has_library_lemma
        )
        expected = sum(1 for _, _, e in tactics_states if e)
        assert included == expected, f"Expected {expected} but got {included}"
