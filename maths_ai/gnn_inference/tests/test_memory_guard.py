from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from maths_ai.gnn_inference.atp_lean_gnn.memory_guard import (
    MemoryGuard,
    MemoryGuardConfig,
    MemoryLimitExceeded,
    MemorySnapshot,
)


class MemoryGuardTests(unittest.TestCase):
    def test_raises_when_process_memory_exceeds_limit(self) -> None:
        guard = MemoryGuard(
            MemoryGuardConfig(
                max_process_memory_gb=1.0,
                max_system_memory_percent=None,
                max_cuda_reserved_percent=None,
            ),
            device=torch.device("cpu"),
        )
        snapshot = MemorySnapshot(
            process_gb=1.5,
            system_percent=20.0,
            cuda_allocated_gb=None,
            cuda_reserved_gb=None,
            cuda_total_gb=None,
        )

        with patch(
            "maths_ai.gnn_inference.atp_lean_gnn.memory_guard.snapshot_memory",
            return_value=snapshot,
        ):
            with self.assertRaisesRegex(MemoryLimitExceeded, "process 1.50GB > 1.00GB"):
                guard.check("unit test")

    def test_disabled_guard_does_not_sample_memory(self) -> None:
        guard = MemoryGuard(
            MemoryGuardConfig(enabled=False),
            device=torch.device("cpu"),
        )

        with patch("maths_ai.gnn_inference.atp_lean_gnn.memory_guard.snapshot_memory") as mock_snapshot:
            self.assertIsNone(guard.check("disabled"))

        mock_snapshot.assert_not_called()

    def test_batch_check_respects_interval_but_checks_first_batch(self) -> None:
        guard = MemoryGuard(
            MemoryGuardConfig(
                max_system_memory_percent=None,
                max_cuda_reserved_percent=None,
                check_every_batches=10,
            ),
            device=torch.device("cpu"),
        )
        snapshot = MemorySnapshot(
            process_gb=0.1,
            system_percent=20.0,
            cuda_allocated_gb=None,
            cuda_reserved_gb=None,
            cuda_total_gb=None,
        )

        with patch(
            "maths_ai.gnn_inference.atp_lean_gnn.memory_guard.snapshot_memory",
            return_value=snapshot,
        ) as mock_snapshot:
            guard.check_batch("first", 1)
            guard.check_batch("middle", 2)
            guard.check_batch("interval", 10)

        self.assertEqual(mock_snapshot.call_count, 2)


if __name__ == "__main__":
    unittest.main()
