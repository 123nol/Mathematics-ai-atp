"""
evaluation/metrics.py
=====================

Recall@k and MRR metric computation for lemma retrieval benchmarks.

Definitions:
    - Recall@k : fraction of queries where the gold library lemma appears
                 in the top-k retrieved results.
    - MRR      : Mean Reciprocal Rank = mean of 1/rank(first gold lemma).
                 Queries with no gold lemma in the full ranked list contribute 0.

No external dependencies beyond stdlib and numpy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class RetrievalResult:
    """Outcome for a single query."""

    query_index: int
    gold_lemma_ids: list[int]          # ground-truth library lemma IDs
    retrieved_lemma_ids: list[int]     # ordered list, ranked 1..N
    hit_rank: int | None               # 1-indexed rank of first gold hit, or None
    excluded: bool = False
    exclusion_reason: str = ""


@dataclass
class MetricSummary:
    """Aggregate metrics over all evaluated queries."""

    target_samples: int                # denominator (queries with library lemma)
    k_values: list[int]
    hit_counts: dict[int, int] = field(default_factory=dict)
    recall: dict[int, float] = field(default_factory=dict)
    mrr: dict[int, float] = field(default_factory=dict)
    median_hit_rank: dict[int, float] = field(default_factory=dict)
    mean_hit_rank: dict[int, float] = field(default_factory=dict)

    # Exclusion breakdown
    excluded_local_hyp: int = 0
    excluded_unresolved: int = 0
    excluded_raw_expr: int = 0
    excluded_no_library_lemma: int = 0
    excluded_other: int = 0

    def to_dict(self) -> dict:
        return {
            "target_samples": self.target_samples,
            "k_values": self.k_values,
            "metrics_by_k": {
                str(k): {
                    "hit_count": self.hit_counts.get(k, 0),
                    "recall": self.recall.get(k, 0.0),
                    "mrr": self.mrr.get(k, 0.0),
                    "median_hit_rank": self.median_hit_rank.get(k, None),
                    "mean_hit_rank": self.mean_hit_rank.get(k, None),
                    "target_samples": self.target_samples,
                }
                for k in self.k_values
            },
            "exclusion_breakdown": {
                "local_hypothesis_only": self.excluded_local_hyp,
                "unresolved_only": self.excluded_unresolved,
                "raw_expression_only": self.excluded_raw_expr,
                "no_library_lemma": self.excluded_no_library_lemma,
                "other": self.excluded_other,
            },
        }

    def format_table(self) -> str:
        """Return a markdown-style table of results."""
        header = "| k | Hit Count | Recall | MRR | Median Hit Rank | Mean Hit Rank |"
        sep    = "| ---: | ---: | ---: | ---: | ---: | ---: |"
        rows   = [header, sep]
        for k in self.k_values:
            hit   = self.hit_counts.get(k, 0)
            rec   = self.recall.get(k, 0.0)
            mrr   = self.mrr.get(k, 0.0)
            med   = self.median_hit_rank.get(k)
            mean  = self.mean_hit_rank.get(k)
            med_s  = f"{med:.2f}" if med is not None else "—"
            mean_s = f"{mean:.2f}" if mean is not None else "—"
            rows.append(
                f"| {k} | {hit} | {rec:.4f} | {mrr:.4f} | {med_s} | {mean_s} |"
            )
        return "\n".join(rows)


def find_first_hit_rank(
    gold_lemma_ids: Sequence[int],
    retrieved_lemma_ids: Sequence[int],
) -> int | None:
    """Return the 1-indexed rank of the first gold lemma in the ranking.

    If no gold lemma appears, returns None.
    """
    gold_set = set(gold_lemma_ids)
    for rank, lid in enumerate(retrieved_lemma_ids, start=1):
        if lid in gold_set:
            return rank
    return None


def compute_metrics(
    results: list[RetrievalResult],
    *,
    k_values: list[int],
) -> MetricSummary:
    """Compute Recall@k and MRR over all non-excluded results.

    Parameters
    ----------
    results : list[RetrievalResult]
        One entry per proof state (excluding states without a library lemma target).
    k_values : list[int]
        The k values to evaluate at (e.g. [200, 500, 1000, 2000, 5000]).

    Returns
    -------
    MetricSummary
    """
    valid = [r for r in results if not r.excluded]
    target_samples = len(valid)

    summary = MetricSummary(target_samples=target_samples, k_values=k_values)

    # Count exclusions
    for r in results:
        if r.excluded:
            reason = r.exclusion_reason
            if reason == "local_hypothesis_only":
                summary.excluded_local_hyp += 1
            elif reason == "unresolved_only":
                summary.excluded_unresolved += 1
            elif reason == "raw_expression_only":
                summary.excluded_raw_expr += 1
            elif reason == "no_library_lemma":
                summary.excluded_no_library_lemma += 1
            else:
                summary.excluded_other += 1

    if not valid:
        # Populate zero values so callers can safely index by k
        for k in k_values:
            summary.hit_counts[k] = 0
            summary.recall[k] = 0.0
            summary.mrr[k] = 0.0
            summary.median_hit_rank[k] = None
            summary.mean_hit_rank[k] = None
        return summary

    hit_ranks = [r.hit_rank for r in valid]  # int or None

    for k in k_values:
        # Recall@k: fraction where hit_rank <= k
        hits_in_k = [r for r in valid if r.hit_rank is not None and r.hit_rank <= k]
        hit_count = len(hits_in_k)
        recall_k  = hit_count / target_samples if target_samples else 0.0

        # MRR at k: count only if hit_rank <= k (or all — mentor uses global MRR)
        mrr_sum = sum(
            1.0 / r.hit_rank
            for r in valid
            if r.hit_rank is not None and r.hit_rank <= k
        )
        mrr_k = mrr_sum / target_samples if target_samples else 0.0

        # Median and mean hit rank over all queries with a hit within k
        hit_ranks_k = [r.hit_rank for r in valid if r.hit_rank is not None and r.hit_rank <= k]
        med_rank = _median(hit_ranks_k) if hit_ranks_k else None
        mean_rank = (sum(hit_ranks_k) / len(hit_ranks_k)) if hit_ranks_k else None

        summary.hit_counts[k] = hit_count
        summary.recall[k] = recall_k
        summary.mrr[k] = mrr_k
        summary.median_hit_rank[k] = med_rank
        summary.mean_hit_rank[k] = mean_rank

    return summary


def compute_global_mrr(results: list[RetrievalResult]) -> float:
    """MRR computed over ALL valid results regardless of k (using best hit rank).

    This matches the mentor's reported MRR=0.0401.
    """
    valid = [r for r in results if not r.excluded]
    if not valid:
        return 0.0
    mrr_sum = sum(
        1.0 / r.hit_rank
        for r in valid
        if r.hit_rank is not None
    )
    return mrr_sum / len(valid)


def _median(values: list[float]) -> float:
    if not values:
        return float("nan")
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
    return float(sorted_vals[mid])
