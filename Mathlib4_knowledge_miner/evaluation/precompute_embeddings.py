#!/usr/bin/env python3
"""
evaluation/precompute_embeddings.py
=====================================

Step 1 of the benchmark: encode all validation proof states with the
lemma-retriever GNN and save to .npz.

CORRECT CHECKPOINT:
    runs/lemma_retriever_clean_v1_50ep_safe/run_20260619_233643/best.pt

Architecture: TacticWithArgsClassifier
Encoding: backbone.encode_nodes() + backbone.readout() → L2 normalize

The output .npz contains:
    embeddings : float32 (N, 512)  — L2 normalized proof-state vectors
    tactics    : object array (N,) — raw tactic strings
    states     : object array (N,) — raw proof state strings
    split      : ["val"]

Requires:
    pip install torch-geometric datasets
    pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \\
        -f https://data.pyg.org/whl/torch-2.10.0+cu128.html

Usage:
    BASE="/path/to/maths_ai_best_improvement_models_20260719"

    python3 -m evaluation.precompute_embeddings \\
        --checkpoint "$BASE/runs/lemma_retriever_clean_v1_50ep_safe/run_20260619_233643/best.pt" \\
        --prepared-root "$BASE/artifacts/prepared/clean_v1" \\
        --output evaluation/precomputed_val_embeddings.npz \\
        --split val
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _check_requirements() -> None:
    errors = []
    try:
        import torch_geometric  # noqa: F401
    except ImportError:
        errors.append(
            "torch_geometric is not installed.\n"
            "  pip install torch-geometric\n"
            "  pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \\\n"
            "      -f https://data.pyg.org/whl/torch-2.10.0+cu128.html"
        )
    try:
        import datasets  # noqa: F401
    except ImportError:
        errors.append("datasets is not installed. Install with:\n  pip install datasets")
    if errors:
        print("\nMISSING DEPENDENCIES:\n")
        for e in errors:
            print(f"  {e}\n")
        sys.exit(1)


# HuggingFace split name mapping
_HF_SPLIT_NAMES = {
    "val":   "validation",
    "test":  "test",
    "train": "train",
}

_DATASET_NAME = "cat-searcher/leandojo-benchmark-4-random"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute lemma-retriever GNN proof-state embeddings for benchmark. "
            "Uses the CORRECT checkpoint: lemma_retriever_clean_v1_50ep_safe."
        )
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "Path to lemma retriever checkpoint. "
            "Must be: runs/lemma_retriever_clean_v1_50ep_safe/run_20260619_233643/best.pt"
        ),
    )
    parser.add_argument(
        "--prepared-root",
        required=True,
        help="Path to artifacts/prepared/clean_v1 (provides vocab files).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for .npz file (e.g. evaluation/precomputed_val_embeddings.npz).",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["val", "test", "train"],
        help="Dataset split to encode. Default: val.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of proof states to encode (for debugging).",
    )
    parser.add_argument(
        "--dataset",
        default=_DATASET_NAME,
        help=f"HuggingFace dataset name. Default: {_DATASET_NAME}",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip L2 normalization (do NOT use — FAISS index requires normalized vectors).",
    )
    args = parser.parse_args(argv)

    _check_requirements()

    # Warn if checkpoint name doesn't look right
    ckpt_path = Path(args.checkpoint)
    if "lemma_retriever" not in str(ckpt_path):
        print(
            "\nWARNING: The checkpoint path does not contain 'lemma_retriever'.\n"
            "  Expected: runs/lemma_retriever_clean_v1_50ep_safe/run_20260619_233643/best.pt\n"
            f"  Got     : {ckpt_path}\n"
            "  Continuing, but results may not reproduce the mentor's audit.\n"
        )

    normalize = not args.no_normalize
    if not normalize:
        print(
            "\nWARNING: --no-normalize was passed. The FAISS index stores normalized vectors.\n"
            "  Query vectors MUST be normalized to reproduce the audit. Proceed with caution.\n"
        )

    print(f"\nLoading GNN encoder from: {args.checkpoint}")
    from evaluation.gnn_encoder import GNNEncoder
    encoder = GNNEncoder.from_checkpoint(
        checkpoint_path=args.checkpoint,
        prepared_root=args.prepared_root,
        normalize=normalize,
    )
    print(f"  Encoder loaded. Device: {encoder.device}, normalize={encoder.normalize}")

    hf_split = _HF_SPLIT_NAMES.get(args.split, args.split)
    print(f"\nStreaming '{args.split}' split ({hf_split}) from: {args.dataset}")
    from datasets import load_dataset
    ds = load_dataset(args.dataset, split=hf_split, streaming=True)

    embeddings_list: list[np.ndarray] = []
    tactics_list: list[str] = []
    states_list: list[str] = []
    failed = 0

    for i, sample in enumerate(ds):
        if args.limit is not None and i >= args.limit:
            break

        state  = str(sample.get("state",  ""))
        tactic = str(sample.get("tactic", ""))

        try:
            emb = encoder.encode(state)
        except Exception:
            if i < 5 or i % 500 == 0:
                print(f"\n  WARNING: Failed to encode sample {i}:")
                traceback.print_exc()
            emb = np.zeros(512, dtype=np.float32)
            failed += 1

        embeddings_list.append(emb)
        tactics_list.append(tactic)
        states_list.append(state)

        if (i + 1) % 200 == 0:
            print(
                f"  Encoded {i + 1} proof states "
                f"(failed={failed}={failed/(i+1)*100:.1f}%)",
                end="\r",
                flush=True,
            )

    total = len(embeddings_list)
    print(f"\n\nEncoded {total} proof states. Failures: {failed} ({failed/max(total,1)*100:.1f}%)")

    if total == 0:
        print("ERROR: No proof states were encoded.")
        return 1

    embeddings  = np.stack(embeddings_list, axis=0)
    tactics_arr = np.array(tactics_list, dtype=object)
    states_arr  = np.array(states_list, dtype=object)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        embeddings=embeddings,
        tactics=tactics_arr,
        states=states_arr,
        split=np.array([args.split]),
        normalize=np.array([normalize]),
    )
    size_mb = out_path.stat().st_size / 1_000_000
    print(f"Saved: {out_path}  ({size_mb:.1f} MB)")
    print(f"  embeddings shape : {embeddings.shape}")
    print(f"  normalize        : {normalize}")
    print()

    # Sanity check: norms should be ~1.0
    if normalize:
        norms = np.linalg.norm(embeddings, axis=1)
        print(f"Norm sanity (should be ~1.0): min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
