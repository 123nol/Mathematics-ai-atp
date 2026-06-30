"""Evaluate a trained premise scorer checkpoint without retraining."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[3]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

from maths_ai.gnn_inference.atp_lean_gnn.lemma_index import LemmaIndex
from maths_ai.gnn_inference.atp_lean_gnn.memory_guard import (
    MemoryLimitExceeded,
    add_memory_guard_args,
    memory_guard_from_args,
)
from maths_ai.gnn_inference.atp_lean_gnn.premise_scoring import PremiseScorer, PremiseScorerConfig
from maths_ai.gnn_inference.atp_lean_gnn.premise_training import evaluate_model_with_premises
from maths_ai.gnn_inference.atp_lean_gnn.reporting import console_print
from maths_ai.gnn_inference.atp_lean_gnn.training import (
    build_dataloaders,
    load_pointer_config,
    load_prepared_metadata,
)
from maths_ai.gnn_inference.scripts.train_scorer import _load_pointer_checkpoint_model


def _load_premise_config(path: Path, *, k_override: int | None = None) -> PremiseScorerConfig:
    with path.open("r", encoding="utf-8") as f:
        config = PremiseScorerConfig(**json.load(f))
    if k_override is None:
        return config
    return PremiseScorerConfig(
        hidden_dim=config.hidden_dim,
        scoring_mode=config.scoring_mode,
        tactic_conditioning=config.tactic_conditioning,
        premise_loss_weight=config.premise_loss_weight,
        k=k_override,
        rerank_size=config.rerank_size,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained premise scorer")
    parser.add_argument("--config", type=str, required=True, help="Path to pointer config")
    parser.add_argument("--premise-config", type=str, required=True, help="Path to premise scoring config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to premise scorer best.pt")
    parser.add_argument("--index-path", type=str, required=True, help="Path to FAISS lemma index directory")
    parser.add_argument("--split", type=str, default="val", choices=("train", "val", "test"))
    parser.add_argument("--output-path", type=str, default=None, help="Optional JSON output path")
    parser.add_argument("--k", type=int, default=None, help="Override number of retrieved lemmas")
    parser.add_argument(
        "--retriever-config",
        type=str,
        default=None,
        help="Optional retriever config for FAISS query embeddings",
    )
    parser.add_argument(
        "--retriever-checkpoint",
        type=str,
        default=None,
        help="Optional retriever checkpoint for FAISS query embeddings",
    )
    parser.add_argument(
        "--freeze-pointer-heads",
        action="store_true",
        help="Report combined loss as premise-only, matching scorer-only experiments",
    )
    add_memory_guard_args(parser)
    args = parser.parse_args(argv)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_amp = device.type == "cuda"
        memory_guard = memory_guard_from_args(args, device=device)

        config = load_pointer_config(Path(args.config))
        metadata = load_prepared_metadata(config.prepared_root)
        premise_config = _load_premise_config(Path(args.premise_config), k_override=args.k)
        checkpoint = torch.load(Path(args.checkpoint), map_location=device, weights_only=False)

        retriever_config_path = args.retriever_config
        retriever_checkpoint_path = args.retriever_checkpoint
        if retriever_config_path is None:
            retriever_config_path = checkpoint.get("retriever_config")
        if retriever_checkpoint_path is None:
            retriever_checkpoint_path = checkpoint.get("retriever_checkpoint")
        if bool(retriever_config_path) != bool(retriever_checkpoint_path):
            raise ValueError("--retriever-config and --retriever-checkpoint must be provided together.")

        console_print(f"Loading lemma index from {args.index_path}...")
        lemma_index = LemmaIndex.load(Path(args.index_path))
        _datasets, loaders = build_dataloaders(metadata, config)

        model = _load_pointer_checkpoint_model(
            metadata=metadata,
            config=config,
            checkpoint_path=Path(args.checkpoint),
            device=device,
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        retriever_model = None
        if retriever_config_path is not None and retriever_checkpoint_path is not None:
            retriever_config = load_pointer_config(Path(retriever_config_path))
            retriever_model = _load_pointer_checkpoint_model(
                metadata=metadata,
                config=retriever_config,
                checkpoint_path=Path(retriever_checkpoint_path),
                device=device,
            )
            for param in retriever_model.parameters():
                param.requires_grad = False
            retriever_model.eval()
            console_print(f"Using retriever query checkpoint: {retriever_checkpoint_path}")

        scorer = PremiseScorer(
            hidden_dim=config.model.hidden_dim,
            mode=premise_config.scoring_mode,
        ).to(device)
        scorer.load_state_dict(checkpoint["scorer_state_dict"])

        metrics = evaluate_model_with_premises(
            model=model,
            scorer=scorer,
            loader=loaders[args.split],
            lemma_index=lemma_index,
            device=device,
            unknown_tactic_id=metadata.unknown_tactic_id,
            arg_loss_weight=config.arg_loss_weight if hasattr(config, "arg_loss_weight") else 0.5,
            premise_loss_weight=premise_config.premise_loss_weight,
            k=premise_config.k,
            split_name=args.split,
            log_every_batches=config.training.log_every_batches,
            use_amp=use_amp,
            pin_memory=config.training.pin_memory,
            memory_guard=memory_guard,
            retrieval_model=retriever_model,
            scorer_only=bool(args.freeze_pointer_heads),
        )

        console_print(
            f"{args.split} | MRR={metrics['premise_mrr']:.4f} | "
            f"Hit@1={metrics['premise_top1_accuracy']:.4f} | "
            f"Hit@5={metrics['premise_top5_accuracy']:.4f} | "
            f"Recall={metrics['premise_recall']:.4f} | "
            f"LocalRecall={metrics['premise_local_recall']:.4f} | "
            f"LemmaRecall={metrics['premise_lemma_recall']:.4f} | "
            f"LemmaRankDelta={metrics['premise_lemma_avg_rank_delta']:.1f}"
        )

        if args.output_path is not None:
            output_path = Path(args.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
            console_print(f"Wrote metrics: {output_path}")
        return 0
    except (FileNotFoundError, MemoryLimitExceeded, RuntimeError, ValueError) as exc:
        console_print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
