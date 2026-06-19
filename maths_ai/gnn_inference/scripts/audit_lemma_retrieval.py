from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[3]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

from maths_ai.gnn_inference.atp_lean_gnn.lemma_index import LemmaIndex
from maths_ai.gnn_inference.atp_lean_gnn.reporting import console_print
from maths_ai.gnn_inference.atp_lean_gnn.training import (
    build_dataloaders,
    build_pointer_model,
    load_pointer_config,
    load_prepared_metadata,
    resolve_device,
)


@dataclass
class RetrievalStats:
    target_samples: int = 0
    hits_by_k: dict[int, int] | None = None
    reciprocal_rank_sum_by_k: dict[int, float] | None = None
    hit_ranks_by_k: dict[int, list[int]] | None = None

    def __post_init__(self) -> None:
        if self.hits_by_k is None:
            self.hits_by_k = {}
        if self.reciprocal_rank_sum_by_k is None:
            self.reciprocal_rank_sum_by_k = {}
        if self.hit_ranks_by_k is None:
            self.hit_ranks_by_k = {}


def _parse_k_values(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        value = int(part.strip())
        if value <= 0:
            raise ValueError("All k values must be positive.")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("At least one k value is required.")
    return sorted(values)


def _load_checkpoint_state_dict(model, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

    adjusted_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(("backbone.", "tactic_embedding.", "argument_selector.")):
            adjusted_state_dict[key] = value
        else:
            adjusted_state_dict[f"backbone.{key}"] = value
    model.load_state_dict(adjusted_state_dict, strict=False)


def _extract_lemma_targets(batch, max_args: int) -> list[list[int]]:
    batch_size = int(batch.y.size(0)) if hasattr(batch, "y") else 1
    per_sample = [[] for _ in range(batch_size)]
    if not (hasattr(batch, "arg_lemma_ids") and hasattr(batch, "arg_count")):
        return per_sample

    flat_targets = batch.arg_lemma_ids.to(device="cpu", dtype=torch.long)
    counts = batch.arg_count.tolist()
    offset = 0
    for sample_index, count in enumerate(counts):
        n_copy = min(int(count), max_args)
        if n_copy > 0:
            values = flat_targets[offset : offset + n_copy].tolist()
            per_sample[sample_index] = [int(value) for value in values if int(value) >= 0]
        offset += int(count)
    return per_sample


def _summarize(stats: RetrievalStats, k_values: list[int]) -> dict[str, object]:
    rows: dict[str, dict[str, object]] = {}
    for k in k_values:
        hits = stats.hits_by_k.get(k, 0)
        ranks = stats.hit_ranks_by_k.get(k, [])
        rows[str(k)] = {
            "target_samples": stats.target_samples,
            "hit_count": hits,
            "recall": 0.0 if stats.target_samples == 0 else hits / stats.target_samples,
            "mrr": (
                0.0
                if stats.target_samples == 0
                else stats.reciprocal_rank_sum_by_k.get(k, 0.0) / stats.target_samples
            ),
            "median_hit_rank": None if not ranks else float(statistics.median(ranks)),
            "mean_hit_rank": None if not ranks else float(statistics.mean(ranks)),
        }
    return rows


def run_audit(
    *,
    config_path: Path,
    checkpoint_path: Path,
    index_path: Path,
    split: str,
    k_values: list[int],
    output_dir: Path | None,
    device_name: str,
) -> dict[str, object]:
    config = load_pointer_config(config_path)
    metadata = load_prepared_metadata(config.prepared_root)
    device = resolve_device(device_name)

    _datasets, loaders = build_dataloaders(metadata, config)
    if split not in loaders:
        raise ValueError(f"Unknown split '{split}'.")

    model = build_pointer_model(metadata, config).to(device)
    _load_checkpoint_state_dict(model, checkpoint_path, device)
    model.eval()

    lemma_index = LemmaIndex.load(index_path)
    max_k = max(k_values)
    stats = RetrievalStats()
    for k in k_values:
        stats.hits_by_k[k] = 0
        stats.reciprocal_rank_sum_by_k[k] = 0.0
        stats.hit_ranks_by_k[k] = []

    console_print(
        f"Auditing lemma retrieval on split '{split}' with k={','.join(str(k) for k in k_values)}..."
    )
    for batch in loaders[split]:
        batch = batch.to(device)
        with torch.no_grad():
            node_embeddings = model.backbone.encode_nodes(batch)
            state_emb = model.backbone.readout(node_embeddings, batch)

        retrieved_ids_batch, _vectors, _scores = lemma_index.search(state_emb, k=max_k)
        target_ids_batch = _extract_lemma_targets(batch, model.max_args)

        for target_ids, retrieved_ids in zip(target_ids_batch, retrieved_ids_batch):
            if not target_ids:
                continue
            stats.target_samples += 1
            target_set = set(target_ids)
            rank = -1
            for index, lemma_id in enumerate(retrieved_ids, start=1):
                if lemma_id in target_set:
                    rank = index
                    break
            if rank < 0:
                continue
            for k in k_values:
                if rank <= k:
                    stats.hits_by_k[k] += 1
                    stats.reciprocal_rank_sum_by_k[k] += 1.0 / rank
                    stats.hit_ranks_by_k[k].append(rank)

    summary = {
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "index_path": str(index_path),
        "prepared_root": str(config.prepared_root),
        "split": split,
        "k_values": k_values,
        "target_samples": stats.target_samples,
        "metrics_by_k": _summarize(stats, k_values),
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "lemma_retrieval_audit.json"
        md_path = output_dir / "lemma_retrieval_audit.md"
        json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        lines = [
            "# Lemma Retrieval Audit",
            "",
            f"- split: `{split}`",
            f"- target samples: `{stats.target_samples}`",
            "",
            "| k | Hit Count | Recall | MRR | Median Hit Rank | Mean Hit Rank |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for k in k_values:
            row = summary["metrics_by_k"][str(k)]
            median = "" if row["median_hit_rank"] is None else f"{row['median_hit_rank']:.2f}"
            mean = "" if row["mean_hit_rank"] is None else f"{row['mean_hit_rank']:.2f}"
            lines.append(
                f"| {k} | {row['hit_count']} | {row['recall']:.4f} | "
                f"{row['mrr']:.4f} | {median} | {mean} |"
            )
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        summary["report_paths"] = {
            "json": str(json_path),
            "markdown": str(md_path),
        }

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit FAISS lemma retrieval recall before premise reranking.")
    parser.add_argument("--config", type=str, required=True, help="Path to pointer config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to pointer checkpoint")
    parser.add_argument("--index-path", type=str, required=True, help="Path to lemma FAISS index directory")
    parser.add_argument("--split", type=str, default="val", help="Prepared split to audit")
    parser.add_argument("--k-values", type=str, default="200,500,1000,2000", help="Comma-separated k values")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional directory for audit reports")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, or cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_audit(
            config_path=Path(args.config),
            checkpoint_path=Path(args.checkpoint),
            index_path=Path(args.index_path),
            split=str(args.split),
            k_values=_parse_k_values(str(args.k_values)),
            output_dir=None if args.output_dir is None else Path(args.output_dir),
            device_name=str(args.device),
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        console_print(f"ERROR: {exc}")
        return 1

    console_print("Lemma retrieval audit:")
    for k, row in summary["metrics_by_k"].items():
        console_print(
            f"  k={k:<5} recall={row['recall']:.4f} "
            f"mrr={row['mrr']:.4f} hits={row['hit_count']}/{row['target_samples']}"
        )
    if "report_paths" in summary:
        console_print(f"Wrote JSON report    : {summary['report_paths']['json']}")
        console_print(f"Wrote Markdown report: {summary['report_paths']['markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
