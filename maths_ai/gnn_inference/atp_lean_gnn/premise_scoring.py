"""Premise scoring head for unified candidate pools.

This module provides ``PremiseScorer``, a tactic-conditioned scoring module
that scores mixed candidate pools (local hypotheses + library lemmas) against
a goal embedding.  Two scoring modes are supported:

- **dot**: Scaled dot-product between a projected query and candidate vectors.
- **source_dot**: Source-aware dot-product with separate local/lemma projections.
- **mlp**: A two-layer MLP that takes the concatenation of query and candidate.

The ``compute_premise_ranking_loss`` function computes cross-entropy ranking
loss over the unified candidate pool for each sample in a batch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .premise_pool import CandidatePool


@dataclass(frozen=True)
class PremiseScorerConfig:
    """Configuration for the premise scoring head."""

    hidden_dim: int = 128
    scoring_mode: str = "dot"  # "dot", "source_dot", or "mlp"
    tactic_conditioning: str = "soft"  # "soft" or "hard"
    premise_loss_weight: float = 0.3
    k: int = 200
    rerank_size: int = 50
    retrieval_score_weight: float = 0.0
    retrieval_score_normalization: str = "zscore"

    def to_dict(self) -> dict[str, object]:
        return {
            "hidden_dim": self.hidden_dim,
            "scoring_mode": self.scoring_mode,
            "tactic_conditioning": self.tactic_conditioning,
            "premise_loss_weight": self.premise_loss_weight,
            "k": self.k,
            "rerank_size": self.rerank_size,
            "retrieval_score_weight": self.retrieval_score_weight,
            "retrieval_score_normalization": self.retrieval_score_normalization,
        }


class PremiseScorer(nn.Module):
    """Score unified candidate premises with tactic-conditioned queries.

    Parameters
    ----------
    hidden_dim : int
        Dimensionality of goal, tactic, and candidate embeddings.
    mode : str
        Scoring mode — ``"dot"`` for scaled dot-product, ``"source_dot"`` for
        source-aware local/lemma dot-product, or ``"mlp"`` for a learned
        two-layer scorer.
    """

    def __init__(
        self,
        hidden_dim: int,
        *,
        mode: str = "dot",
        retrieval_score_weight: float = 0.0,
        retrieval_score_normalization: str = "zscore",
    ) -> None:
        super().__init__()

        if mode not in {"dot", "source_dot", "mlp"}:
            raise ValueError(f"Unsupported scoring mode '{mode}'. Use 'dot', 'source_dot', or 'mlp'.")
        if retrieval_score_normalization not in {"raw", "zscore", "minmax"}:
            raise ValueError(
                "Unsupported retrieval_score_normalization "
                f"'{retrieval_score_normalization}'. Use 'raw', 'zscore', or 'minmax'."
            )

        self.mode = mode
        self.hidden_dim = hidden_dim
        self.retrieval_score_weight = float(retrieval_score_weight)
        self.retrieval_score_normalization = retrieval_score_normalization

        if mode == "source_dot":
            self.local_query_proj = nn.Linear(hidden_dim * 2, hidden_dim)
            self.local_key_proj = nn.Linear(hidden_dim, hidden_dim)
            self.lemma_query_proj = nn.Linear(hidden_dim * 2, hidden_dim)
            self.lemma_key_proj = nn.Linear(hidden_dim, hidden_dim)
            self.query_proj = None
            self.key_proj = None
        else:
            # Project [goal_vec; tactic_emb] -> hidden_dim
            self.query_proj = nn.Linear(hidden_dim * 2, hidden_dim)
            self.key_proj = nn.Linear(hidden_dim, hidden_dim)
            self.local_query_proj = None
            self.local_key_proj = None
            self.lemma_query_proj = None
            self.lemma_key_proj = None

        if mode == "mlp":
            self.scorer = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.scorer = None

        self._scale = 1.0 / math.sqrt(hidden_dim)

    def score(
        self,
        goal_vec: Tensor,
        tactic_emb: Tensor,
        candidate_vectors: Tensor,
        *,
        candidate_sources: list[str] | None = None,
        candidate_retrieval_scores: list[float | None] | None = None,
        retrieval_goal_vec: Tensor | None = None,
    ) -> Tensor:
        """Score each candidate against the tactic-conditioned goal.

        Parameters
        ----------
        goal_vec : Tensor
            Goal embedding, shape ``[H]`` or ``[1, H]``.
        tactic_emb : Tensor
            Tactic embedding, shape ``[H]`` or ``[1, H]``.
        candidate_vectors : Tensor
            Candidate embeddings, shape ``[C, H]``.
        candidate_sources : list[str] | None
            Candidate source labels, used by ``source_dot`` mode.
        candidate_retrieval_scores : list[float | None] | None
            Optional retriever scores aligned to candidates. ``None`` entries
            are ignored, which is expected for local hypotheses.
        retrieval_goal_vec : Tensor | None
            Retriever-space goal embedding for lemma candidates in
            ``source_dot`` mode.

        Returns
        -------
        Tensor
            Scores, shape ``[C]``.
        """
        # Flatten to [H]
        goal = goal_vec.view(-1)
        tactic = tactic_emb.view(-1)

        if self.mode == "source_dot":
            if candidate_sources is None:
                raise ValueError("candidate_sources are required for source_dot scoring.")
            lemma_goal = goal if retrieval_goal_vec is None else retrieval_goal_vec.view(-1)
            scores = candidate_vectors.new_empty(candidate_vectors.size(0))

            local_indices = [idx for idx, source in enumerate(candidate_sources) if source == "local"]
            if local_indices:
                index = torch.tensor(local_indices, dtype=torch.long, device=candidate_vectors.device)
                query = self.local_query_proj(torch.cat([goal, tactic], dim=0))
                keys = self.local_key_proj(candidate_vectors.index_select(0, index))
                scores.index_copy_(0, index, (keys @ query) * self._scale)

            lemma_indices = [idx for idx, source in enumerate(candidate_sources) if source == "lemma"]
            if lemma_indices:
                index = torch.tensor(lemma_indices, dtype=torch.long, device=candidate_vectors.device)
                query = self.lemma_query_proj(torch.cat([lemma_goal, tactic], dim=0))
                keys = self.lemma_key_proj(candidate_vectors.index_select(0, index))
                scores.index_copy_(0, index, (keys @ query) * self._scale)

            return self._add_retrieval_score_residual(
                scores,
                candidate_retrieval_scores=candidate_retrieval_scores,
            )

        query = self.query_proj(torch.cat([goal, tactic], dim=0))
        candidate_vectors = self.key_proj(candidate_vectors)

        if self.mode == "dot":
            # Scaled dot-product
            scores = (candidate_vectors @ query) * self._scale  # [C]
        else:
            # MLP: concat query with each candidate and score
            num_candidates = candidate_vectors.size(0)
            query_expanded = query.unsqueeze(0).expand(num_candidates, -1)  # [C, H]
            combined = torch.cat([query_expanded, candidate_vectors], dim=1)  # [C, 2H]
            scores = self.scorer(combined).squeeze(-1)  # [C]

        return self._add_retrieval_score_residual(
            scores,
            candidate_retrieval_scores=candidate_retrieval_scores,
        )

    def _add_retrieval_score_residual(
        self,
        scores: Tensor,
        *,
        candidate_retrieval_scores: list[float | None] | None,
    ) -> Tensor:
        """Add a non-trainable retriever residual to lemma candidates."""
        if self.retrieval_score_weight == 0.0 or not candidate_retrieval_scores:
            return scores

        scored_indices: list[int] = []
        retrieval_values: list[float] = []
        for idx, retrieval_score in enumerate(candidate_retrieval_scores):
            if retrieval_score is not None:
                scored_indices.append(idx)
                retrieval_values.append(float(retrieval_score))

        if not scored_indices:
            return scores

        values = torch.tensor(
            retrieval_values,
            dtype=scores.dtype,
            device=scores.device,
        )
        if self.retrieval_score_normalization == "zscore":
            values = values - values.mean()
            if values.numel() > 1:
                values = values / values.std(unbiased=False).clamp_min(1e-6)
        elif self.retrieval_score_normalization == "minmax":
            min_value = values.min()
            span = (values.max() - min_value).clamp_min(1e-6)
            values = (values - min_value) / span

        residual = scores.new_zeros(scores.shape)
        index = torch.tensor(scored_indices, dtype=torch.long, device=scores.device)
        residual.index_copy_(0, index, values)
        return scores + self.retrieval_score_weight * residual

    def forward(
        self,
        goal_vecs: Tensor,
        tactic_embs: Tensor,
        pools: list[CandidatePool],
        *,
        retrieval_goal_vecs: Tensor | None = None,
    ) -> list[Tensor]:
        """Score all candidate pools in a batch.

        Parameters
        ----------
        goal_vecs : Tensor
            Goal embeddings, shape ``[B, H]``.
        tactic_embs : Tensor
            Tactic embeddings, shape ``[B, H]``.
        pools : list[CandidatePool]
            One pool per sample in the batch.
        retrieval_goal_vecs : Tensor | None
            Retriever-space goal embeddings, shape ``[B, H]``. Used by
            ``source_dot`` mode for lemma candidates.

        Returns
        -------
        list[Tensor]
            Per-sample score tensors, each of shape ``[C_i]``.
        """
        batch_size = goal_vecs.size(0)
        if len(pools) != batch_size:
            raise ValueError(
                f"Number of pools ({len(pools)}) does not match "
                f"batch size ({batch_size})."
            )

        all_scores: list[Tensor] = []
        for b in range(batch_size):
            scores = self.score(
                goal_vecs[b],
                tactic_embs[b],
                pools[b].candidate_vectors,
                candidate_sources=pools[b].candidate_sources,
                candidate_retrieval_scores=pools[b].candidate_retrieval_scores,
                retrieval_goal_vec=None if retrieval_goal_vecs is None else retrieval_goal_vecs[b],
            )
            all_scores.append(scores)

        return all_scores


def _find_target_index_in_pool(
    pool: CandidatePool,
    *,
    arg_node_indices: list[int],
    arg_lemma_ids: list[int],
) -> int:
    """Find the index of the true premise in the candidate pool.

    Priority:
    1. If any ``arg_node_indices`` entry is >= 0 and matches a local candidate,
       return the pool position of that local node.
    2. If any ``arg_lemma_ids`` entry is >= 0 and matches a library candidate,
       return the pool position of that lemma.
    3. Return -1 if no match is found.
    """
    target_idx, _source = _find_primary_target_in_pool(
        pool,
        arg_node_indices=arg_node_indices,
        arg_lemma_ids=arg_lemma_ids,
    )
    return target_idx


def _find_primary_target_in_pool(
    pool: CandidatePool,
    *,
    arg_node_indices: list[int],
    arg_lemma_ids: list[int],
) -> tuple[int, str]:
    """Find the primary target and report whether it is local or library.

    If an example has both local and lemma targets, local is treated as the
    primary target to match the pointer model's historical target priority.
    """
    local_targets = [node_id for node_id in arg_node_indices if node_id >= 0]
    if local_targets:
        for node_id in local_targets:
            for pool_idx, (source, cid) in enumerate(
                zip(pool.candidate_sources, pool.candidate_ids)
            ):
                if source == "local" and cid == node_id:
                    return pool_idx, "local"
        return -1, "local"

    lemma_targets = [lemma_id for lemma_id in arg_lemma_ids if lemma_id >= 0]
    if lemma_targets:
        for lemma_id in lemma_targets:
            for pool_idx, (source, cid) in enumerate(
                zip(pool.candidate_sources, pool.candidate_ids)
            ):
                if source == "lemma" and cid == lemma_id:
                    return pool_idx, "lemma"
        return -1, "lemma"

    return -1, "none"


def compute_premise_ranking_loss(
    score_list: list[Tensor],
    pools: list[CandidatePool],
    arg_node_indices: Tensor,
    arg_lemma_ids: Tensor,
) -> tuple[Tensor, dict[str, float]]:
    """Cross-entropy ranking loss over unified candidate pools, with metrics.

    For each sample in the batch, we find the true premise in the candidate
    pool and compute a cross-entropy loss against the scored candidates.
    Also tracks retrieval and reranking metrics.

    Parameters
    ----------
    score_list : list[Tensor]
        Per-sample score tensors from ``PremiseScorer.forward()``.
    pools : list[CandidatePool]
        One pool per sample.
    arg_node_indices : Tensor
        Ground-truth local node indices, shape ``[B, max_args]``, -1 for invalid.
    arg_lemma_ids : Tensor
        Ground-truth lemma IDs, shape ``[B, max_args]``, -1 for invalid.

    Returns
    -------
    loss : Tensor
        Scalar ranking loss (averaged over valid samples).
    metrics : dict
        ``"premise_loss"``, ``"valid_samples"``, ``"total_samples"``,
        ``"target_present_count"``, ``"top1_correct"``, ``"top5_correct"``,
        ``"mrr_sum"``.
    """
    batch_size = len(score_list)
    device = score_list[0].device if score_list else torch.device("cpu")

    losses: list[Tensor] = []
    valid_count = 0
    target_present_count = 0
    top1_correct = 0
    top5_correct = 0
    mrr_sum = 0.0
    source_metrics = {
        "local": {
            "target": 0,
            "valid": 0,
            "top1": 0,
            "top5": 0,
            "mrr_sum": 0.0,
            "rerank_comparable": 0,
            "rerank_improved": 0,
            "rerank_worsened": 0,
            "rerank_unchanged": 0,
            "retrieval_rank_sum": 0.0,
            "scorer_rank_sum": 0.0,
            "rank_delta_sum": 0.0,
        },
        "lemma": {
            "target": 0,
            "valid": 0,
            "top1": 0,
            "top5": 0,
            "mrr_sum": 0.0,
            "rerank_comparable": 0,
            "rerank_improved": 0,
            "rerank_worsened": 0,
            "rerank_unchanged": 0,
            "retrieval_rank_sum": 0.0,
            "scorer_rank_sum": 0.0,
            "rank_delta_sum": 0.0,
        },
    }
    rerank_comparable_count = 0
    rerank_improved_count = 0
    rerank_worsened_count = 0
    rerank_unchanged_count = 0
    rerank_retrieval_rank_sum = 0.0
    rerank_scorer_rank_sum = 0.0
    rerank_delta_sum = 0.0

    for b in range(batch_size):
        scores = score_list[b]  # [C_b]
        pool = pools[b]

        # Get ground-truth node/lemma IDs for this sample
        b_node_ids = arg_node_indices[b].tolist() if arg_node_indices.dim() > 1 else [int(arg_node_indices[b].item())]
        b_lemma_ids = arg_lemma_ids[b].tolist() if arg_lemma_ids.dim() > 1 else [int(arg_lemma_ids[b].item())]

        has_target = any(i >= 0 for i in b_node_ids) or any(i >= 0 for i in b_lemma_ids)
        if not has_target:
            continue

        target_idx, target_source = _find_primary_target_in_pool(
            pool,
            arg_node_indices=b_node_ids,
            arg_lemma_ids=b_lemma_ids,
        )
        target_present_count += 1
        if target_source in source_metrics:
            source_metrics[target_source]["target"] += 1

        if target_idx < 0:
            # Target exists but wasn't retrieved in the pool — skip loss
            continue

        target = torch.tensor(target_idx, dtype=torch.long, device=device)
        loss = F.cross_entropy(scores.unsqueeze(0), target.unsqueeze(0))
        losses.append(loss)
        valid_count += 1

        # Reranking metrics
        # Sort scores in descending order to find the rank of the true target
        sorted_indices = scores.argsort(descending=True).tolist()
        rank = sorted_indices.index(target_idx) + 1  # 1-indexed

        if rank == 1:
            top1_correct += 1
        if rank <= 5:
            top5_correct += 1
        mrr_sum += 1.0 / rank
        if target_source in source_metrics:
            source_metrics[target_source]["valid"] += 1
            if rank == 1:
                source_metrics[target_source]["top1"] += 1
            if rank <= 5:
                source_metrics[target_source]["top5"] += 1
            source_metrics[target_source]["mrr_sum"] += 1.0 / rank

        retrieval_rank = None
        if pool.candidate_retrieval_ranks:
            retrieval_rank = pool.candidate_retrieval_ranks[target_idx]

        if retrieval_rank is not None:
            rank_delta = float(retrieval_rank - rank)
            rerank_comparable_count += 1
            rerank_retrieval_rank_sum += float(retrieval_rank)
            rerank_scorer_rank_sum += float(rank)
            rerank_delta_sum += rank_delta
            if rank < retrieval_rank:
                rerank_improved_count += 1
            elif rank > retrieval_rank:
                rerank_worsened_count += 1
            else:
                rerank_unchanged_count += 1

            if target_source in source_metrics:
                source = source_metrics[target_source]
                source["rerank_comparable"] += 1
                source["retrieval_rank_sum"] += float(retrieval_rank)
                source["scorer_rank_sum"] += float(rank)
                source["rank_delta_sum"] += rank_delta
                if rank < retrieval_rank:
                    source["rerank_improved"] += 1
                elif rank > retrieval_rank:
                    source["rerank_worsened"] += 1
                else:
                    source["rerank_unchanged"] += 1

    if losses:
        total_loss = torch.stack(losses).mean()
    else:
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)

    metrics = {
        "premise_loss": float(total_loss.item()),
        "valid_samples": valid_count,
        "total_samples": batch_size,
        "target_present_count": target_present_count,
        "top1_correct": top1_correct,
        "top5_correct": top5_correct,
        "mrr_sum": mrr_sum,
        "rerank_comparable_count": rerank_comparable_count,
        "rerank_improved_count": rerank_improved_count,
        "rerank_worsened_count": rerank_worsened_count,
        "rerank_unchanged_count": rerank_unchanged_count,
        "rerank_retrieval_rank_sum": rerank_retrieval_rank_sum,
        "rerank_scorer_rank_sum": rerank_scorer_rank_sum,
        "rerank_delta_sum": rerank_delta_sum,
        "local_target_count": source_metrics["local"]["target"],
        "local_valid_samples": source_metrics["local"]["valid"],
        "local_top1_correct": source_metrics["local"]["top1"],
        "local_top5_correct": source_metrics["local"]["top5"],
        "local_mrr_sum": source_metrics["local"]["mrr_sum"],
        "local_rerank_comparable_count": source_metrics["local"]["rerank_comparable"],
        "local_rerank_improved_count": source_metrics["local"]["rerank_improved"],
        "local_rerank_worsened_count": source_metrics["local"]["rerank_worsened"],
        "local_rerank_unchanged_count": source_metrics["local"]["rerank_unchanged"],
        "local_rerank_retrieval_rank_sum": source_metrics["local"]["retrieval_rank_sum"],
        "local_rerank_scorer_rank_sum": source_metrics["local"]["scorer_rank_sum"],
        "local_rerank_delta_sum": source_metrics["local"]["rank_delta_sum"],
        "lemma_target_count": source_metrics["lemma"]["target"],
        "lemma_valid_samples": source_metrics["lemma"]["valid"],
        "lemma_top1_correct": source_metrics["lemma"]["top1"],
        "lemma_top5_correct": source_metrics["lemma"]["top5"],
        "lemma_mrr_sum": source_metrics["lemma"]["mrr_sum"],
        "lemma_rerank_comparable_count": source_metrics["lemma"]["rerank_comparable"],
        "lemma_rerank_improved_count": source_metrics["lemma"]["rerank_improved"],
        "lemma_rerank_worsened_count": source_metrics["lemma"]["rerank_worsened"],
        "lemma_rerank_unchanged_count": source_metrics["lemma"]["rerank_unchanged"],
        "lemma_rerank_retrieval_rank_sum": source_metrics["lemma"]["retrieval_rank_sum"],
        "lemma_rerank_scorer_rank_sum": source_metrics["lemma"]["scorer_rank_sum"],
        "lemma_rerank_delta_sum": source_metrics["lemma"]["rank_delta_sum"],
    }

    return total_loss, metrics
