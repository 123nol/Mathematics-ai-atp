from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch
from torch import Tensor


class LemmaIndexLike(Protocol):
    def search(
        self,
        state_vecs: Tensor,
        *,
        k: int,
    ) -> tuple[list[list[int]], np.ndarray, np.ndarray]:
        ...


@dataclass(frozen=True)
class CandidatePool:
    candidate_vectors: Tensor
    candidate_sources: list[str]
    candidate_ids: list[int]
    local_node_ids: list[int]
    lemma_ids: list[int]
    candidate_retrieval_ranks: list[int | None] = field(default_factory=list)
    candidate_retrieval_scores: list[float | None] = field(default_factory=list)


def build_unified_pools(
    state_vecs: Tensor,
    node_embeddings: Tensor,
    premise_mask: Tensor,
    batch_index: Tensor,
    *,
    lemma_index: LemmaIndexLike,
    k: int = 500,
    retrieval_state_vecs: Tensor | None = None,
) -> list[CandidatePool]:
    """Return per-graph candidate pools combining local and library premises."""
    if state_vecs.dim() != 2:
        raise ValueError("state_vecs must be [batch, hidden_dim].")
    if retrieval_state_vecs is not None and retrieval_state_vecs.dim() != 2:
        raise ValueError("retrieval_state_vecs must be [batch, hidden_dim].")
    if node_embeddings.dim() != 2:
        raise ValueError("node_embeddings must be [total_nodes, hidden_dim].")

    device = node_embeddings.device
    batch_size = int(state_vecs.size(0))
    search_vecs = state_vecs if retrieval_state_vecs is None else retrieval_state_vecs
    if int(search_vecs.size(0)) != batch_size:
        raise ValueError("retrieval_state_vecs must have the same batch size as state_vecs.")

    lemma_ids_batch, lemma_vecs_batch, retrieval_scores_batch = lemma_index.search(search_vecs, k=k)
    if len(lemma_ids_batch) != batch_size:
        raise ValueError("lemma_index returned a batch size mismatch.")

    lemma_vecs_batch = torch.from_numpy(lemma_vecs_batch).to(device=device, dtype=node_embeddings.dtype)

    pools: list[CandidatePool] = []
    for b in range(batch_size):
        graph_mask = batch_index == b
        graph_node_ids = graph_mask.nonzero(as_tuple=False).view(-1)
        graph_offset = int(graph_node_ids[0].item()) if graph_node_ids.numel() > 0 else 0

        local_mask = graph_mask & premise_mask
        local_ids = local_mask.nonzero(as_tuple=False).view(-1)
        local_vecs = node_embeddings.index_select(0, local_ids)
        local_id_list = [int(i - graph_offset) for i in local_ids.tolist()]

        lemma_ids = [int(x) for x in lemma_ids_batch[b]]
        lemma_vecs = lemma_vecs_batch[b]
        lemma_ranks: list[int | None] = list(range(1, len(lemma_ids) + 1))
        lemma_scores: list[float | None] = [
            float(score) for score in retrieval_scores_batch[b][: len(lemma_ids)]
        ]

        if local_vecs.numel() == 0:
            candidate_vectors = lemma_vecs
            candidate_sources = ["lemma"] * len(lemma_ids)
            candidate_ids = lemma_ids
            candidate_retrieval_ranks = lemma_ranks
            candidate_retrieval_scores = lemma_scores
        elif lemma_vecs.numel() == 0:
            candidate_vectors = local_vecs
            candidate_sources = ["local"] * len(local_id_list)
            candidate_ids = local_id_list
            candidate_retrieval_ranks = [None] * len(local_id_list)
            candidate_retrieval_scores = [None] * len(local_id_list)
        else:
            candidate_vectors = torch.cat([local_vecs, lemma_vecs], dim=0)
            candidate_sources = ["local"] * len(local_id_list) + ["lemma"] * len(lemma_ids)
            candidate_ids = local_id_list + lemma_ids
            candidate_retrieval_ranks = [None] * len(local_id_list) + lemma_ranks
            candidate_retrieval_scores = [None] * len(local_id_list) + lemma_scores

        pools.append(
            CandidatePool(
                candidate_vectors=candidate_vectors,
                candidate_sources=candidate_sources,
                candidate_ids=candidate_ids,
                local_node_ids=local_id_list,
                lemma_ids=lemma_ids,
                candidate_retrieval_ranks=candidate_retrieval_ranks,
                candidate_retrieval_scores=candidate_retrieval_scores,
            )
        )

    return pools
