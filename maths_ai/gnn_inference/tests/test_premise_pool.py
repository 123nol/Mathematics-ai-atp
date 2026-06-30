from __future__ import annotations

import unittest

import numpy as np
import torch

from maths_ai.gnn_inference.atp_lean_gnn.premise_pool import build_unified_pools


class _FakeLemmaIndex:
    def __init__(self) -> None:
        self.last_goal_vecs = None

    def search(self, goal_vecs, *, k):
        self.last_goal_vecs = goal_vecs.detach().clone()
        batch_size = int(goal_vecs.size(0))
        dim = int(goal_vecs.size(1))
        lemma_ids = []
        lemma_vecs = []
        for b in range(batch_size):
            lemma_ids.append([100 + 2 * b, 100 + 2 * b + 1])
            lemma_vecs.append(
                np.full((2, dim), fill_value=float(b + 1), dtype=np.float32)
            )
        lemma_vecs = np.stack(lemma_vecs, axis=0)
        scores = np.array(
            [[0.9 - 0.1 * i for i in range(2)] for _ in range(batch_size)],
            dtype=np.float32,
        )
        return lemma_ids, lemma_vecs, scores


class PremisePoolTests(unittest.TestCase):
    def test_builds_unified_pool(self) -> None:
        goal_vecs = torch.randn(2, 4)
        node_embeddings = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
            ],
            dtype=torch.float,
        )
        premise_mask = torch.tensor([True, False, True, False, True, True])
        batch_index = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

        pools = build_unified_pools(
            goal_vecs,
            node_embeddings,
            premise_mask,
            batch_index,
            lemma_index=_FakeLemmaIndex(),
            k=2,
        )

        self.assertEqual(len(pools), 2)

        pool0 = pools[0]
        self.assertEqual(pool0.local_node_ids, [0, 2])
        self.assertEqual(pool0.lemma_ids, [100, 101])
        self.assertEqual(pool0.candidate_sources.count("local"), 2)
        self.assertEqual(pool0.candidate_sources.count("lemma"), 2)
        self.assertEqual(pool0.candidate_retrieval_ranks, [None, None, 1, 2])
        self.assertIsNone(pool0.candidate_retrieval_scores[0])
        self.assertIsNone(pool0.candidate_retrieval_scores[1])
        self.assertAlmostEqual(pool0.candidate_retrieval_scores[2], 0.9)
        self.assertAlmostEqual(pool0.candidate_retrieval_scores[3], 0.8)
        self.assertEqual(pool0.candidate_vectors.shape[0], 4)

        pool1 = pools[1]
        self.assertEqual(pool1.local_node_ids, [1, 2])
        self.assertEqual(pool1.lemma_ids, [102, 103])
        self.assertEqual(pool1.candidate_sources.count("local"), 2)
        self.assertEqual(pool1.candidate_sources.count("lemma"), 2)
        self.assertEqual(pool1.candidate_retrieval_ranks, [None, None, 1, 2])
        self.assertIsNone(pool1.candidate_retrieval_scores[0])
        self.assertIsNone(pool1.candidate_retrieval_scores[1])
        self.assertAlmostEqual(pool1.candidate_retrieval_scores[2], 0.9)
        self.assertAlmostEqual(pool1.candidate_retrieval_scores[3], 0.8)
        self.assertEqual(pool1.candidate_vectors.shape[0], 4)

    def test_uses_retrieval_state_vectors_for_lemma_search(self) -> None:
        goal_vecs = torch.zeros(2, 4)
        retrieval_vecs = torch.ones(2, 4)
        node_embeddings = torch.randn(4, 4)
        premise_mask = torch.tensor([True, False, True, False])
        batch_index = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        lemma_index = _FakeLemmaIndex()

        build_unified_pools(
            goal_vecs,
            node_embeddings,
            premise_mask,
            batch_index,
            lemma_index=lemma_index,
            k=2,
            retrieval_state_vecs=retrieval_vecs,
        )

        self.assertTrue(torch.equal(lemma_index.last_goal_vecs, retrieval_vecs))
