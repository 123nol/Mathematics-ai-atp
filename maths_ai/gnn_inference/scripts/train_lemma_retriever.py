from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch_geometric.data import Batch, Data

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[3]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

from maths_ai.gnn_inference.atp_lean_gnn.graph import lemma_statement_to_dag
from maths_ai.gnn_inference.atp_lean_gnn.lemma_index import LemmaIndex
from maths_ai.gnn_inference.atp_lean_gnn.lemma_corpus import LemmaRecord, load_lemma_corpus
from maths_ai.gnn_inference.atp_lean_gnn.logger import TrainingLogger
from maths_ai.gnn_inference.atp_lean_gnn.pyg import dag_to_pyg
from maths_ai.gnn_inference.atp_lean_gnn.reporting import console_print
from maths_ai.gnn_inference.atp_lean_gnn.training import (
    build_dataloaders,
    build_pointer_model,
    load_pointer_config,
    load_prepared_metadata,
    resolve_device,
    transform_edge_index,
)


@dataclass(frozen=True)
class RetrieverBatch:
    state_sample_indices: torch.Tensor
    lemma_batch: Batch
    labels: torch.Tensor
    positive_count: int


def _lemma_text(record: LemmaRecord, *, mode: str) -> str:
    if mode == "statement":
        return record.statement
    if mode == "name_statement":
        return f"{record.name} {record.statement}"
    raise ValueError("lemma text mode must be 'statement' or 'name_statement'.")


class LemmaGraphCache:
    def __init__(
        self,
        *,
        records: list[LemmaRecord],
        node_vocab: dict[str, int],
        edge_mode: str,
        lemma_text_mode: str,
        max_cache_entries: int,
    ) -> None:
        self.records_by_id = {record.lemma_id: record for record in records}
        self.lemma_ids = list(self.records_by_id)
        self.node_vocab = node_vocab
        self.edge_mode = edge_mode
        self.lemma_text_mode = lemma_text_mode
        self.max_cache_entries = max_cache_entries
        self._cache: OrderedDict[int, Data | None] = OrderedDict()

    def get(self, lemma_id: int) -> Data | None:
        if lemma_id in self._cache:
            value = self._cache.pop(lemma_id)
            self._cache[lemma_id] = value
            return value

        record = self.records_by_id.get(lemma_id)
        if record is None:
            self._remember(lemma_id, None)
            return None

        try:
            dag = lemma_statement_to_dag(_lemma_text(record, mode=self.lemma_text_mode))
            data = dag_to_pyg(dag, self.node_vocab)
            state_ids = [node.id for node in dag.nodes if node.label == "State"]
            if not state_ids:
                raise ValueError("Lemma graph is missing State node.")
            data.state_node_index = torch.tensor([state_ids[-1]], dtype=torch.long)
            data.edge_index = transform_edge_index(data.edge_index, edge_mode=self.edge_mode)
        except Exception:
            data = None
        self._remember(lemma_id, data)
        return data

    def _remember(self, lemma_id: int, data: Data | None) -> None:
        if self.max_cache_entries <= 0:
            return
        self._cache[lemma_id] = data
        while len(self._cache) > self.max_cache_entries:
            self._cache.popitem(last=False)

    def sample_negatives(self, *, count: int, excluded: set[int]) -> list[int]:
        if count <= 0:
            return []
        candidates = [lemma_id for lemma_id in self.lemma_ids if lemma_id not in excluded]
        if not candidates:
            return []
        if len(candidates) <= count:
            return candidates
        return random.sample(candidates, count)


def _create_run_dir(run_root: Path) -> Path:
    run_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = run_root / f"run_{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = run_root / f"run_{timestamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def _format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining_seconds = divmod(seconds, 60)
    return f"{int(minutes)}m {remaining_seconds:.0f}s"


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
    targets: list[list[int]] = [[] for _ in range(batch_size)]
    if not (hasattr(batch, "arg_lemma_ids") and hasattr(batch, "arg_count")):
        return targets

    flat_targets = batch.arg_lemma_ids.to(device="cpu", dtype=torch.long)
    counts = batch.arg_count.tolist()
    offset = 0
    for sample_index, count in enumerate(counts):
        n_copy = min(int(count), max_args)
        if n_copy > 0:
            values = flat_targets[offset : offset + n_copy].tolist()
            targets[sample_index] = [int(value) for value in values if int(value) >= 0]
        offset += int(count)
    return targets


def _mine_hard_negatives(
    *,
    model,
    batch,
    lemma_index: LemmaIndex | None,
    targets_by_sample: list[list[int]],
    hard_negatives: int,
    max_hard_negatives_per_batch: int,
) -> list[int]:
    if lemma_index is None or hard_negatives <= 0 or max_hard_negatives_per_batch <= 0:
        return []

    with torch.no_grad():
        node_embeddings = model.backbone.encode_nodes(batch)
        state_emb = model.backbone.readout(node_embeddings, batch)
    retrieved_ids_batch, _vectors, _scores = lemma_index.search(
        state_emb,
        k=hard_negatives + max((len(targets) for targets in targets_by_sample), default=0) + 8,
    )

    hard_ids: list[int] = []
    seen: set[int] = set()
    for target_ids, retrieved_ids in zip(targets_by_sample, retrieved_ids_batch):
        excluded = set(target_ids)
        added_for_sample = 0
        for lemma_id in retrieved_ids:
            lemma_id = int(lemma_id)
            if lemma_id < 0 or lemma_id in excluded or lemma_id in seen:
                continue
            hard_ids.append(lemma_id)
            seen.add(lemma_id)
            added_for_sample += 1
            if added_for_sample >= hard_negatives or len(hard_ids) >= max_hard_negatives_per_batch:
                break
        if len(hard_ids) >= max_hard_negatives_per_batch:
            break
    return hard_ids


def _build_retriever_batch(
    *,
    batch,
    graph_cache: LemmaGraphCache,
    max_args: int,
    random_negatives: int,
    hard_negative_ids: list[int],
    max_lemma_candidates: int,
    device: torch.device,
) -> RetrieverBatch | None:
    targets_by_sample = _extract_lemma_targets(batch, max_args)
    state_indices: list[int] = []
    positive_lemma_ids: list[int] = []
    lemma_data_list: list[Data] = []

    for sample_index, lemma_ids in enumerate(targets_by_sample):
        for lemma_id in lemma_ids:
            lemma_data = graph_cache.get(lemma_id)
            if lemma_data is None:
                continue
            state_indices.append(sample_index)
            positive_lemma_ids.append(lemma_id)
            lemma_data_list.append(lemma_data)

    if len(state_indices) < 2:
        return None

    positive_set = set(positive_lemma_ids)
    negative_budget = max(max_lemma_candidates - len(lemma_data_list), 0)
    added_negatives = 0
    for negative_id in hard_negative_ids:
        if added_negatives >= negative_budget:
            break
        if negative_id in positive_set:
            continue
        lemma_data = graph_cache.get(negative_id)
        if lemma_data is not None:
            lemma_data_list.append(lemma_data)
            added_negatives += 1

    remaining_random = min(random_negatives, max(negative_budget - added_negatives, 0))
    for negative_id in graph_cache.sample_negatives(count=remaining_random, excluded=positive_set):
        lemma_data = graph_cache.get(negative_id)
        if lemma_data is not None:
            lemma_data_list.append(lemma_data)

    if len(lemma_data_list) < len(state_indices):
        return None

    lemma_batch = Batch.from_data_list(lemma_data_list).to(device)
    return RetrieverBatch(
        state_sample_indices=torch.tensor(state_indices, dtype=torch.long, device=device),
        lemma_batch=lemma_batch,
        labels=torch.arange(len(state_indices), dtype=torch.long, device=device),
        positive_count=len(state_indices),
    )


def _contrastive_step(
    *,
    model,
    state_batch,
    retriever_batch: RetrieverBatch,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    state_node_embeddings = model.backbone.encode_nodes(state_batch)
    state_emb = model.backbone.readout(state_node_embeddings, state_batch)
    state_emb = state_emb.index_select(0, retriever_batch.state_sample_indices)

    lemma_node_embeddings = model.backbone.encode_nodes(retriever_batch.lemma_batch)
    lemma_emb = model.backbone.readout(lemma_node_embeddings, retriever_batch.lemma_batch)

    state_emb = F.normalize(state_emb, dim=1)
    lemma_emb = F.normalize(lemma_emb, dim=1)
    logits = state_emb @ lemma_emb.t()
    logits = logits / temperature
    loss = F.cross_entropy(logits, retriever_batch.labels)

    with torch.no_grad():
        predictions = logits.argmax(dim=1)
        top1_correct = int((predictions == retriever_batch.labels).sum().item())
        top_k = min(5, logits.size(1))
        topk = logits.topk(top_k, dim=1).indices
        top5_correct = int((topk == retriever_batch.labels.unsqueeze(1)).any(dim=1).sum().item())

    return loss, {
        "loss": float(loss.item()),
        "positive_count": retriever_batch.positive_count,
        "candidate_count": int(logits.size(1)),
        "top1_correct": top1_correct,
        "top5_correct": top5_correct,
    }


def _run_epoch(
    *,
    model,
    loader,
    graph_cache: LemmaGraphCache,
    optimizer,
    grad_scaler,
    device: torch.device,
    max_args: int,
    random_negatives: int,
    hard_negative_index: LemmaIndex | None,
    hard_negatives: int,
    max_hard_negatives_per_batch: int,
    max_lemma_candidates: int,
    temperature: float,
    grad_clip: float,
    train: bool,
    epoch: int,
    total_epochs: int,
    log_every_batches: int,
    use_amp: bool,
) -> dict[str, float | int]:
    model.train(mode=train)
    total_loss = 0.0
    total_positive = 0
    total_top1 = 0
    total_top5 = 0
    used_batches = 0
    skipped_batches = 0
    total_candidate_count = 0
    start_time = time.perf_counter()
    phase = "train" if train else "val"
    console_print(f"  Starting {phase} epoch {epoch:02d}/{total_epochs:02d}...")

    for batch_index, batch in enumerate(loader, start=1):
        batch = batch.to(device)
        targets_by_sample = _extract_lemma_targets(batch, max_args)
        hard_negative_ids = _mine_hard_negatives(
            model=model,
            batch=batch,
            lemma_index=hard_negative_index if train else None,
            targets_by_sample=targets_by_sample,
            hard_negatives=hard_negatives if train else 0,
            max_hard_negatives_per_batch=max_hard_negatives_per_batch if train else 0,
        )
        retriever_batch = _build_retriever_batch(
            batch=batch,
            graph_cache=graph_cache,
            max_args=max_args,
            random_negatives=random_negatives if train else 0,
            hard_negative_ids=hard_negative_ids,
            max_lemma_candidates=max_lemma_candidates,
            device=device,
        )
        if retriever_batch is None:
            skipped_batches += 1
            continue

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                loss, metrics = _contrastive_step(
                    model=model,
                    state_batch=batch,
                    retriever_batch=retriever_batch,
                    temperature=temperature,
                )

        if train:
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.backbone.parameters(), grad_clip)
            grad_scaler.step(optimizer)
            grad_scaler.update()

        positives = int(metrics["positive_count"])
        total_loss += float(metrics["loss"]) * positives
        total_positive += positives
        total_top1 += int(metrics["top1_correct"])
        total_top5 += int(metrics["top5_correct"])
        total_candidate_count += int(metrics["candidate_count"])
        used_batches += 1

        if batch_index == 1 or batch_index % log_every_batches == 0 or batch_index == len(loader):
            elapsed = _format_elapsed(time.perf_counter() - start_time)
            console_print(
                f"    {phase} batch {batch_index:>5}/{len(loader)} | "
                f"targets={total_positive} | "
                f"loss={total_loss / max(total_positive, 1):.4f} | "
                f"top1={total_top1 / max(total_positive, 1):.4f} | "
                f"top5={total_top5 / max(total_positive, 1):.4f} | "
                f"elapsed={elapsed}"
            )

    return {
        "loss": total_loss / max(total_positive, 1),
        "top1_accuracy": total_top1 / max(total_positive, 1),
        "top5_accuracy": total_top5 / max(total_positive, 1),
        "positive_count": total_positive,
        "used_batches": used_batches,
        "skipped_batches": skipped_batches,
        "mean_candidate_count": total_candidate_count / max(used_batches, 1),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train state-to-lemma contrastive retriever embeddings.")
    parser.add_argument("--config", type=str, required=True, help="Path to clean pointer config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to clean pointer checkpoint")
    parser.add_argument("--lemma-corpus", type=str, required=True, help="Path to lemmas.jsonl")
    parser.add_argument("--run-root", type=str, default="runs/lemma_retriever", help="Directory for retriever runs")
    parser.add_argument("--epochs", type=int, default=5, help="Number of contrastive training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Backbone learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Optimizer weight decay")
    parser.add_argument("--temperature", type=float, default=0.07, help="Contrastive softmax temperature")
    parser.add_argument("--random-negatives", type=int, default=256, help="Random lemma negatives per batch")
    parser.add_argument("--hard-negative-index", type=str, default=None, help="Optional FAISS index used to mine hard negative lemmas")
    parser.add_argument("--hard-negatives", type=int, default=0, help="Hard negatives mined per lemma-target sample")
    parser.add_argument("--max-hard-negatives-per-batch", type=int, default=128, help="Maximum mined hard negatives added to one GPU batch")
    parser.add_argument("--max-lemma-candidates-per-batch", type=int, default=512, help="Maximum lemma graphs encoded in one contrastive batch")
    parser.add_argument("--lemma-cache-size", type=int, default=4096, help="Maximum cached CPU lemma graphs; 0 disables caching")
    parser.add_argument(
        "--lemma-text-mode",
        type=str,
        default="statement",
        choices=("statement", "name_statement"),
        help="How to build lemma graphs for retriever training",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--log-every-batches", type=int, default=100, help="Batch logging interval")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.epochs < 1:
        console_print("ERROR: --epochs must be positive")
        return 1
    if args.temperature <= 0:
        console_print("ERROR: --temperature must be positive")
        return 1

    config = load_pointer_config(Path(args.config), epochs_override=args.epochs)
    metadata = load_prepared_metadata(config.prepared_root)
    device = resolve_device(str(args.device))
    use_amp = device.type == "cuda"

    run_dir = _create_run_dir(Path(args.run_root))
    console_print(f"Saving retriever run to {run_dir}")

    model = build_pointer_model(metadata, config).to(device)
    _load_checkpoint_state_dict(model, Path(args.checkpoint), device)

    records = load_lemma_corpus(args.lemma_corpus)
    graph_cache = LemmaGraphCache(
        records=records,
        node_vocab=metadata.node_vocab,
        edge_mode=config.edge_mode,
        lemma_text_mode=str(args.lemma_text_mode),
        max_cache_entries=int(args.lemma_cache_size),
    )
    hard_negative_index = (
        None
        if args.hard_negative_index is None
        else LemmaIndex.load(Path(args.hard_negative_index))
    )

    _datasets, loaders = build_dataloaders(metadata, config)
    optimizer = AdamW(
        model.backbone.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    grad_scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    logger = TrainingLogger(run_dir)
    best_val_top1 = -1.0

    config_payload = {
        "base_config": config.to_dict(),
        "checkpoint": str(args.checkpoint),
        "lemma_corpus": str(args.lemma_corpus),
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "temperature": float(args.temperature),
        "random_negatives": int(args.random_negatives),
        "hard_negative_index": None if args.hard_negative_index is None else str(args.hard_negative_index),
        "hard_negatives": int(args.hard_negatives),
        "max_hard_negatives_per_batch": int(args.max_hard_negatives_per_batch),
        "max_lemma_candidates_per_batch": int(args.max_lemma_candidates_per_batch),
        "lemma_cache_size": int(args.lemma_cache_size),
        "lemma_text_mode": str(args.lemma_text_mode),
    }
    (run_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    (run_dir / "retriever_config.json").write_text(
        json.dumps(config_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = _run_epoch(
            model=model,
            loader=loaders["train"],
            graph_cache=graph_cache,
            optimizer=optimizer,
            grad_scaler=grad_scaler,
            device=device,
            max_args=model.max_args,
            random_negatives=int(args.random_negatives),
            hard_negative_index=hard_negative_index,
            hard_negatives=int(args.hard_negatives),
            max_hard_negatives_per_batch=int(args.max_hard_negatives_per_batch),
            max_lemma_candidates=int(args.max_lemma_candidates_per_batch),
            temperature=float(args.temperature),
            grad_clip=float(args.grad_clip),
            train=True,
            epoch=epoch,
            total_epochs=int(args.epochs),
            log_every_batches=int(args.log_every_batches),
            use_amp=use_amp,
        )
        val_metrics = _run_epoch(
            model=model,
            loader=loaders["val"],
            graph_cache=graph_cache,
            optimizer=optimizer,
            grad_scaler=grad_scaler,
            device=device,
            max_args=model.max_args,
            random_negatives=0,
            hard_negative_index=None,
            hard_negatives=0,
            max_hard_negatives_per_batch=0,
            max_lemma_candidates=int(args.max_lemma_candidates_per_batch),
            temperature=float(args.temperature),
            grad_clip=float(args.grad_clip),
            train=False,
            epoch=epoch,
            total_epochs=int(args.epochs),
            log_every_batches=int(args.log_every_batches),
            use_amp=use_amp,
        )

        console_print(
            f"Epoch {epoch} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_top1={val_metrics['top1_accuracy']:.4f} | "
            f"val_top5={val_metrics['top5_accuracy']:.4f}"
        )
        if float(val_metrics["top1_accuracy"]) > best_val_top1:
            best_val_top1 = float(val_metrics["top1_accuracy"])
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "config": config.to_dict(),
                    "retriever_config": config_payload,
                    "val_metrics": val_metrics,
                },
                run_dir / "best.pt",
            )

        logger.log_epoch(
            epoch,
            {
                "train_loss": float(train_metrics["loss"]),
                "train_top1_accuracy": float(train_metrics["top1_accuracy"]),
                "train_top5_accuracy": float(train_metrics["top5_accuracy"]),
                "train_positive_count": int(train_metrics["positive_count"]),
                "train_used_batches": int(train_metrics["used_batches"]),
                "train_skipped_batches": int(train_metrics["skipped_batches"]),
                "train_mean_candidate_count": float(train_metrics["mean_candidate_count"]),
                "val_loss": float(val_metrics["loss"]),
                "val_top1_accuracy": float(val_metrics["top1_accuracy"]),
                "val_top5_accuracy": float(val_metrics["top5_accuracy"]),
                "val_positive_count": int(val_metrics["positive_count"]),
                "val_used_batches": int(val_metrics["used_batches"]),
                "val_skipped_batches": int(val_metrics["skipped_batches"]),
                "val_mean_candidate_count": float(val_metrics["mean_candidate_count"]),
                "best_val_top1": best_val_top1,
            },
        )

    console_print(f"Best checkpoint: {run_dir / 'best.pt'}")
    console_print(f"Learning curves saved to {logger.jsonl_path} and {logger.csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
