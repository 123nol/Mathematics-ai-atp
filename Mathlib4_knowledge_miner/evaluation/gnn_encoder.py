"""
evaluation/gnn_encoder.py
=========================

GNN proof-state encoder for the lemma retrieval benchmark.

This module wraps the CORRECT lemma-retriever checkpoint:
    runs/lemma_retriever_clean_v1_50ep_safe/run_20260619_233643/best.pt

Architecture: TacticWithArgsClassifier (NOT GraphSAGEStateClassifier directly)
    - backbone: GraphSAGEStateClassifier (4 SAGEConv layers, hidden_dim=512)
    - tactic_embedding: nn.Embedding(228, 512)
    - argument_selector: ArgumentSelector(512)

Query encoding path (matching build_lemma_index.py):
    batch → model.backbone.encode_nodes(batch) → node_embeddings
         → model.backbone.readout(node_embeddings, batch) → state_emb (512-dim)
         → L2 normalize → query vector

IMPORTANT:
    - Do NOT use pointer_gnn_clean_v1/best.pt as the encoder.
    - The tactic_embedding and argument_selector heads are NOT used for retrieval.
    - Both lemma vectors and query vectors must be L2-normalized (inner product = cosine).

Requires: torch_geometric, datasets

Install:
    pip install torch-geometric datasets
    pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
        -f https://data.pyg.org/whl/torch-2.10.0+cu128.html
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

_TORCH_GEOMETRIC_MISSING = """
torch_geometric is not installed. Install with:
    pip install torch-geometric
    pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \\
        -f https://data.pyg.org/whl/torch-2.10.0+cu128.html
"""


def _require_torch_geometric() -> None:
    try:
        import torch_geometric  # noqa: F401
    except ImportError as exc:
        raise ImportError(_TORCH_GEOMETRIC_MISSING) from exc


def _add_ref_repo_to_path() -> None:
    import os
    ref_repo = Path(os.environ.get("MATH_ATP_DIR", "../Mathematics-ai-atp")).resolve()
    if ref_repo.exists():
        ref_str = str(ref_repo)
        if ref_str not in sys.path:
            sys.path.insert(0, ref_str)


def _normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalize a 1D or 2D array row-wise."""
    if v.ndim == 1:
        norm = np.linalg.norm(v)
        return (v / norm) if norm > 1e-12 else v
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return v / norms


class GNNEncoder:
    """Encode Lean proof states into 512-dimensional L2-normalized embeddings.

    Uses the lemma-retriever-fine-tuned checkpoint:
        lemma_retriever_clean_v1_50ep_safe/run_20260619_233643/best.pt

    Architecture: TacticWithArgsClassifier
        backbone.encode_nodes(batch) → backbone.readout(node_embs, batch) → L2 normalize

    Parameters
    ----------
    model : TacticWithArgsClassifier loaded from the lemma retriever checkpoint
    node_vocab : dict[str, int] from prepared/clean_v1/vocab/node_vocab.json
    device : torch.device
    edge_mode : "bidirectional" (matches training config)
    normalize : bool — must be True to match FAISS index (default True)
    """

    def __init__(
        self,
        model,
        node_vocab: dict[str, int],
        device,
        *,
        edge_mode: str = "bidirectional",
        normalize: bool = True,
    ) -> None:
        _require_torch_geometric()
        self.model = model
        self.node_vocab = node_vocab
        self.device = device
        self.edge_mode = edge_mode
        self.normalize = normalize
        self._state_label_id = node_vocab.get("State", 0)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        prepared_root: str | Path,
        *,
        normalize: bool = True,
    ) -> "GNNEncoder":
        """Load the encoder from the lemma retriever checkpoint.

        Parameters
        ----------
        checkpoint_path : path to lemma_retriever_clean_v1_50ep_safe/.../best.pt
        prepared_root   : path to artifacts/prepared/clean_v1
        normalize       : L2-normalize output (must be True to match FAISS index)
        """
        _require_torch_geometric()
        _add_ref_repo_to_path()

        import torch

        checkpoint_path = Path(checkpoint_path)
        prepared_root = Path(prepared_root)

        # --- Load checkpoint ---
        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

        # --- Read architecture config from checkpoint ---
        # The checkpoint embeds its full config under 'config'
        cfg = ckpt.get("config", {})
        model_cfg = cfg.get("model", {})
        hidden_dim  = int(model_cfg.get("hidden_dim", 512))
        num_layers  = int(model_cfg.get("num_layers", 4))
        dropout     = float(model_cfg.get("dropout", 0.2))
        max_args    = int(model_cfg.get("max_args", 3))
        edge_mode   = str(cfg.get("edge_mode", "bidirectional"))
        use_node_type = bool(cfg.get("use_node_type", True))

        # --- Load vocab ---
        node_vocab_path = prepared_root / "vocab" / "node_vocab.json"
        tactic_vocab_path = prepared_root / "vocab" / "tactic_vocab.json"
        node_vocab   = json.loads(node_vocab_path.read_text(encoding="utf-8"))
        tactic_vocab = json.loads(tactic_vocab_path.read_text(encoding="utf-8"))

        num_node_labels = len(node_vocab)
        num_tactics     = len(tactic_vocab)

        # Verify against checkpoint shapes
        sd = ckpt["model_state_dict"]
        ckpt_num_labels  = sd["backbone.label_embedding.weight"].shape[0]
        ckpt_num_tactics = sd["backbone.classifier.weight"].shape[0]
        ckpt_hidden_dim  = sd["backbone.label_embedding.weight"].shape[1]

        if num_node_labels != ckpt_num_labels:
            raise ValueError(
                f"Vocab size mismatch: vocab has {num_node_labels} labels, "
                f"but checkpoint has {ckpt_num_labels}. "
                f"Use the prepared_root that matches this checkpoint."
            )
        if num_tactics != ckpt_num_tactics:
            raise ValueError(
                f"Tactic vocab size mismatch: vocab has {num_tactics}, "
                f"checkpoint has {ckpt_num_tactics}."
            )
        if hidden_dim != ckpt_hidden_dim:
            hidden_dim = ckpt_hidden_dim  # trust checkpoint

        # Infer num_node_types from checkpoint
        num_node_types = sd["backbone.node_type_embedding.weight"].shape[0]

        # --- Build model ---
        from maths_ai.gnn_inference.atp_lean_gnn.argument_selector import (
            TacticWithArgsClassifier,
        )

        model = TacticWithArgsClassifier(
            num_node_labels=num_node_labels,
            num_tactics=num_tactics,
            num_node_types=num_node_types,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_node_type=use_node_type,
            max_args=max_args,
        )

        missing, unexpected = model.load_state_dict(sd, strict=True), []
        # strict=True raises on mismatch, so if we're here weights loaded cleanly

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        return cls(model, node_vocab, device, edge_mode=edge_mode, normalize=normalize)

    # ------------------------------------------------------------------
    # Edge index transformation (must match training)
    # ------------------------------------------------------------------

    def _transform_edge_index(self, edge_index):
        """Apply bidirectional edge duplication matching training config."""
        import torch
        if self.edge_mode != "bidirectional" or edge_index.numel() == 0:
            return edge_index.to(dtype=torch.long).contiguous()
        forward = edge_index.to(dtype=torch.long)
        reverse = forward[[1, 0], :]
        combined = torch.cat([forward, reverse], dim=1)
        return torch.unique(combined, dim=1).contiguous()

    def _infer_state_node_index(self, data) -> "torch.Tensor":
        """Locate the State node — mirrors training.py:infer_state_node_index."""
        import torch
        state_label_id = self._state_label_id
        state_matches = (data.x == state_label_id).nonzero(as_tuple=False).view(-1)
        if state_matches.numel() == 0:
            # Fallback: last node (degenerate DAG)
            return torch.tensor([data.x.size(0) - 1], dtype=torch.long)

        if data.edge_index.numel() > 0:
            source_nodes = set(data.edge_index[0].tolist())
            root_candidates = [int(n) for n in state_matches.tolist() if n not in source_nodes]
            if len(root_candidates) == 1:
                return torch.tensor(root_candidates, dtype=torch.long)

        # Single match fallback
        return state_matches[:1].to(dtype=torch.long)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, proof_state: str) -> np.ndarray:
        """Encode one proof state string → 512-dim L2-normalized float32 vector.

        This is the EXACT encoding used to generate the FAISS index queries.

        Parameters
        ----------
        proof_state : str
            Raw Lean proof state (e.g. "h : Nat ⊢ 0 < h").

        Returns
        -------
        np.ndarray, shape (512,), dtype float32, L2-normalized
        """
        _add_ref_repo_to_path()
        import torch
        from torch_geometric.data import Batch

        from maths_ai.gnn_inference.atp_lean_gnn.graph import proof_state_to_dag
        from maths_ai.gnn_inference.atp_lean_gnn.pyg import dag_to_pyg

        dag  = proof_state_to_dag(proof_state)
        data = dag_to_pyg(dag, self.node_vocab)

        data.state_node_index = self._infer_state_node_index(data)
        data.edge_index = self._transform_edge_index(data.edge_index)
        data = data.to(self.device)

        batch = Batch.from_data_list([data])

        with torch.no_grad():
            node_embs = self.model.backbone.encode_nodes(batch)
            state_emb = self.model.backbone.readout(node_embs, batch)

        vec = state_emb.squeeze(0).detach().cpu().numpy().astype(np.float32)
        if self.normalize:
            vec = _normalize(vec)
        return vec

    def encode_batch(self, proof_states: list[str]) -> np.ndarray:
        """Encode a list of proof states → (N, 512) float32 array.

        Note: currently encodes one at a time (variable-size graphs).
        For large batches, use precompute_embeddings.py which handles errors.
        """
        return np.stack([self.encode(s) for s in proof_states], axis=0)
