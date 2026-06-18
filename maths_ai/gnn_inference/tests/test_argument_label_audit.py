from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from maths_ai.gnn_inference.atp_lean_gnn.argument_audit import (
    ArgumentAuditConfig,
    run_argument_label_audit,
)
from maths_ai.gnn_inference.atp_lean_gnn.argument_labels import (
    GRAPH_NON_CANDIDATE,
    LOCAL_HYPOTHESIS,
    RAW_EXPRESSION,
    UNRESOLVED,
)
from maths_ai.gnn_inference.atp_lean_gnn.dataset import DatasetRow


class ArgumentLabelAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.output_root = Path("tests") / "_tmp_argument_audit"
        if self.output_root.exists():
            shutil.rmtree(self.output_root)

    def tearDown(self) -> None:
        if self.output_root.exists():
            shutil.rmtree(self.output_root)

    def _row(self, *, state: str, tactic: str, row_index: int) -> DatasetRow:
        return DatasetRow(
            state=state,
            theorem=f"demo.{row_index}",
            tactic=tactic,
            split="train",
            row_index=row_index,
            dataset_name="fake/dataset",
        )

    @patch("maths_ai.gnn_inference.atp_lean_gnn.argument_audit.iter_dataset_rows")
    def test_argument_audit_writes_resolution_report(self, mock_iter_dataset_rows) -> None:
        rows = [
            self._row(state="h : P x\n⊢ P x", tactic="exact h", row_index=0),
            self._row(state="⊢ P x", tactic="exact x", row_index=1),
            self._row(state="⊢ P x", tactic="exact P x", row_index=2),
            self._row(state="⊢ P x", tactic="rw [missing_lemma]", row_index=3),
        ]

        def fake_iter_dataset_rows(*, dataset_name: str, split: str, sample_limit: int | None = None):
            return iter(rows if sample_limit is None else rows[:sample_limit])

        mock_iter_dataset_rows.side_effect = fake_iter_dataset_rows

        summary = run_argument_label_audit(
            ArgumentAuditConfig(
                dataset_name="fake/dataset",
                splits=("train",),
                output_root=self.output_root,
                force=True,
            )
        )

        categories = summary["overall"]["category_counts"]
        self.assertEqual(categories[LOCAL_HYPOTHESIS], 1)
        self.assertGreaterEqual(categories[GRAPH_NON_CANDIDATE], 1)
        self.assertEqual(categories[UNRESOLVED], 1)
        self.assertEqual(categories[RAW_EXPRESSION], 1)

        json_path = self.output_root / "reports" / "argument_label_audit.json"
        md_path = self.output_root / "reports" / "argument_label_audit.md"
        self.assertTrue(json_path.exists())
        self.assertTrue(md_path.exists())

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertIn("tactic_category_counts", payload["overall"])
        markdown = md_path.read_text(encoding="utf-8")
        self.assertIn("Argument Label Audit", markdown)
        self.assertIn("Representative Noisy Examples", markdown)


if __name__ == "__main__":
    unittest.main()
