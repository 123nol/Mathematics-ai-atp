from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


from .cache import (
    SplitReport,
    append_failure_record,
    build_failure_record,
    build_json_payload,
    build_summary,
    prepare_output_root,
    write_json_artifact,
    write_manifest,
    write_pyg_artifact,
    write_summary_json,
    write_summary_markdown,
    write_vocab,
)
from .dataset import DATASET_NAME, canonicalize_split_name, iter_dataset_rows
from .preparation import prepare_example
from .argument_labels import (
    LIBRARY_LEMMA,
    LOCAL_HYPOTHESIS,
    RAW_EXPRESSION,
    ArgumentLabelAnalysis,
    analyze_argument_labels,
)
from .labels import build_tactic_vocab, encode_tactic_name
from .lemma_corpus import load_lemma_name_index
from .pyg import build_vocab_from_labels, dag_to_pyg
from .reporting import console_print


DEFAULT_OUTPUT_ROOT = Path("artifacts") / "prepared" / "v1"


@dataclass(frozen=True)
class PreprocessConfig:
    dataset_name: str = DATASET_NAME
    splits: tuple[str, ...] = ("train", "val", "test")
    output_root: Path = DEFAULT_OUTPUT_ROOT
    sample_per_split: int | None = None
    lemma_corpus_path: Path | None = None
    force: bool = False


def _clean_argument_targets(
    analysis: ArgumentLabelAnalysis,
) -> tuple[list[int], list[int], dict[str, object]]:
    category_counts = Counter(resolution.category for resolution in analysis.resolutions)
    if analysis.has_raw_expression_argument:
        category_counts[RAW_EXPRESSION] += 1

    if analysis.has_raw_expression_argument:
        skipped_arguments = len(analysis.resolutions)
        return [], [], {
            "total_arguments": len(analysis.resolutions),
            "trainable_local": 0,
            "trainable_lemma": 0,
            "skipped_arguments": skipped_arguments,
            "has_raw_expression": True,
            "category_counts": category_counts,
        }

    arg_indices: list[int] = []
    arg_lemma_ids: list[int] = []
    for resolution in analysis.resolutions:
        if resolution.category == LOCAL_HYPOTHESIS:
            arg_indices.append(resolution.node_id)
            arg_lemma_ids.append(-1)
        elif resolution.category == LIBRARY_LEMMA:
            arg_indices.append(-1)
            arg_lemma_ids.append(resolution.lemma_id)

    trainable_local = sum(1 for node_id in arg_indices if node_id >= 0)
    trainable_lemma = sum(1 for lemma_id in arg_lemma_ids if lemma_id >= 0)
    skipped_arguments = len(analysis.resolutions) - trainable_local - trainable_lemma
    return arg_indices, arg_lemma_ids, {
        "total_arguments": len(analysis.resolutions),
        "trainable_local": trainable_local,
        "trainable_lemma": trainable_lemma,
        "skipped_arguments": skipped_arguments,
        "has_raw_expression": False,
        "category_counts": category_counts,
    }


def _normalize_splits(raw_splits: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(raw_splits, str):
        candidates = [part.strip() for part in raw_splits.split(",")]
    else:
        candidates = [part.strip() for part in raw_splits]

    splits: list[str] = []
    for split in candidates:
        if not split:
            continue
        canonical_split = canonicalize_split_name(split)
        if canonical_split not in splits:
            splits.append(canonical_split)

    if not splits:
        raise ValueError("At least one split must be provided.")
    if "train" not in splits:
        raise ValueError("The requested splits must include 'train' so train-only vocabularies can be built.")
    return ["train", *[split for split in splits if split != "train"]]


def scan_train_split(
    *,
    dataset_name: str,
    sample_per_split: int | None,
) -> tuple[dict[str, int], dict[str, int], SplitReport]:
    node_labels: set[str] = set()
    tactic_names: list[str] = []
    report = SplitReport(split="train")

    for row in iter_dataset_rows(
        dataset_name=dataset_name,
        split="train",
        sample_limit=sample_per_split,
    ):
        try:
            example = prepare_example(row)
        except Exception as exc:
            failure_record = build_failure_record(row, exc)
            report.record_failure(
                category=str(failure_record["failure_category"]),
                phase=str(failure_record["phase"]),
            )
            continue

        report.record_success(dag=example.dag, tactic_name=example.tactic_name)
        node_labels.update(node.label for node in example.dag.nodes)
        tactic_names.append(example.tactic_name)

    if report.success_count == 0:
        raise RuntimeError("The train split produced zero successful examples while building vocabularies.")

    node_vocab = build_vocab_from_labels(node_labels)
    tactic_vocab = build_tactic_vocab(tactic_names)
    return node_vocab, tactic_vocab, report


def process_split(
    *,
    dataset_name: str,
    split: str,
    sample_per_split: int | None,
    output_root: Path,
    node_vocab: dict[str, int],
    tactic_vocab: dict[str, int],
    lemma_name_index: dict[str, int] | None,
) -> tuple[SplitReport, dict[str, object]]:
    import torch

    from .pyg import build_premise_mask

    report = SplitReport(split=split)
    for row in iter_dataset_rows(
        dataset_name=dataset_name,
        split=split,
        sample_limit=sample_per_split,
    ):
        try:
            example = prepare_example(row)
        except Exception as exc:
            failure_record = build_failure_record(row, exc)
            append_failure_record(output_root, split=split, record=failure_record)
            report.record_failure(
                category=str(failure_record["failure_category"]),
                phase=str(failure_record["phase"]),
            )
            continue

        json_payload = build_json_payload(
            example.row,
            parsed_state=example.parsed_state,
            dag=example.dag,
            tactic_name=example.tactic_name,
        )
        write_json_artifact(
            output_root,
            split=split,
            row_index=example.row.row_index,
            payload=json_payload,
        )

        data = dag_to_pyg(example.dag, node_vocab)
        data.y = torch.tensor(
            [encode_tactic_name(example.tactic_name, tactic_vocab)],
            dtype=torch.long,
        )
        data.split = split
        data.row_index = example.row.row_index
        data.dataset_name = example.row.dataset_name
        data.theorem = example.row.theorem
        data.tactic_raw = example.row.tactic
        data.tactic_name = example.tactic_name

        # --- Argument-selection ground truth (additive) ---------------
        premise_mask = build_premise_mask(example.dag)
        data.premise_mask = torch.tensor(premise_mask, dtype=torch.bool)

        argument_analysis = analyze_argument_labels(
            raw_tactic=example.row.tactic,
            dag=example.dag,
            lemma_name_index=lemma_name_index,
            premise_mask=premise_mask,
        )
        arg_indices, arg_lemma_ids, argument_label_stats = _clean_argument_targets(argument_analysis)
        data.arg_node_indices = torch.tensor(arg_indices, dtype=torch.long) if arg_indices else torch.tensor([], dtype=torch.long)
        data.arg_lemma_ids = torch.tensor(arg_lemma_ids, dtype=torch.long) if arg_lemma_ids else torch.tensor([], dtype=torch.long)
        data.arg_count = len(arg_indices)
        data.arg_raw_expression = bool(argument_label_stats["has_raw_expression"])
        data.arg_total_parsed_count = int(argument_label_stats["total_arguments"])
        data.arg_trainable_count = len(arg_indices)
        data.arg_skipped_count = int(argument_label_stats["skipped_arguments"])
        # --------------------------------------------------------------

        write_pyg_artifact(
            output_root,
            split=split,
            row_index=example.row.row_index,
            data=data,
        )

        report.record_success(dag=example.dag, tactic_name=example.tactic_name)
        report.record_argument_labels(
            total_arguments=int(argument_label_stats["total_arguments"]),
            trainable_local=int(argument_label_stats["trainable_local"]),
            trainable_lemma=int(argument_label_stats["trainable_lemma"]),
            skipped_arguments=int(argument_label_stats["skipped_arguments"]),
            has_raw_expression=bool(argument_label_stats["has_raw_expression"]),
            category_counts=argument_label_stats["category_counts"],
        )

    if report.success_count == 0:
        raise RuntimeError(f"Split '{split}' produced zero successful examples.")

    manifest = report.to_manifest(
        dataset_name=dataset_name,
        output_root=output_root,
        vocab_source="train",
        sample_limit=sample_per_split,
    )
    write_manifest(output_root, split=split, manifest=manifest)
    return report, manifest


def run_preprocessing(config: PreprocessConfig) -> dict[str, object]:
    output_root = Path(config.output_root)
    if output_root.exists() and not config.force:
        raise FileExistsError(
            f"Output root '{output_root}' already exists. Re-run with --force to overwrite it."
        )

    console_print(
        f"\n  Scanning train split from {config.dataset_name} to build train-only vocabularies..."
    )
    node_vocab, tactic_vocab, train_scan = scan_train_split(
        dataset_name=config.dataset_name,
        sample_per_split=config.sample_per_split,
    )
    console_print(
        f"  Train scan complete: attempted={train_scan.attempted_count}, "
        f"success={train_scan.success_count}, failure={train_scan.failure_count}"
    )

    lemma_name_index = None
    if config.lemma_corpus_path is not None:
        lemma_name_index = load_lemma_name_index(config.lemma_corpus_path)

    prepare_output_root(output_root, splits=list(config.splits), force=config.force)
    write_vocab(output_root, name="node_vocab.json", vocab=node_vocab)
    write_vocab(output_root, name="tactic_vocab.json", vocab=tactic_vocab)

    split_reports: dict[str, SplitReport] = {}
    manifests: dict[str, dict[str, object]] = {}
    for split in config.splits:
        console_print(f"\n  Processing split '{split}'...")
        report, manifest = process_split(
            dataset_name=config.dataset_name,
            split=split,
            sample_per_split=config.sample_per_split,
            output_root=output_root,
            node_vocab=node_vocab,
            tactic_vocab=tactic_vocab,
            lemma_name_index=lemma_name_index,
        )
        split_reports[split] = report
        manifests[split] = manifest
        console_print(
            f"  Finished '{split}': attempted={report.attempted_count}, "
            f"success={report.success_count}, failure={report.failure_count}"
        )

    summary = build_summary(
        dataset_name=config.dataset_name,
        output_root=output_root,
        splits=list(config.splits),
        manifests=manifests,
        split_reports=split_reports,
        node_vocab=node_vocab,
        tactic_vocab=tactic_vocab,
    )
    summary_json_path = write_summary_json(output_root, summary)
    summary_md_path = write_summary_markdown(output_root, summary)

    console_print(f"\n  Wrote node vocab     : {output_root / 'vocab' / 'node_vocab.json'}")
    console_print(f"  Wrote tactic vocab   : {output_root / 'vocab' / 'tactic_vocab.json'}")
    console_print(f"  Wrote JSON summary   : {summary_json_path}")
    console_print(f"  Wrote Markdown summary: {summary_md_path}")

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare cached graph artifacts from LeanDojo proof states")
    parser.add_argument("--dataset-name", type=str, default=DATASET_NAME, help="Dataset name to stream from Hugging Face")
    parser.add_argument("--splits", type=str, default="train,val,test", help="Comma-separated splits to preprocess (must include train)")
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT), help="Output directory for prepared artifacts")
    parser.add_argument("--sample-per-split", type=int, default=None, help="Optional limit of examples to process per split")
    parser.add_argument("--lemma-corpus", type=str, default=None, help="Optional lemma corpus JSONL for library premise labels")
    parser.add_argument("--force", action="store_true", help="Overwrite the output root if it already exists")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        config = PreprocessConfig(
            dataset_name=args.dataset_name,
            splits=tuple(_normalize_splits(args.splits)),
            output_root=Path(args.output_root),
            sample_per_split=args.sample_per_split,
            lemma_corpus_path=None if args.lemma_corpus is None else Path(args.lemma_corpus),
            force=args.force,
        )
        run_preprocessing(config)
    except (FileExistsError, RuntimeError, ValueError) as exc:
        console_print(f"  ERROR: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
