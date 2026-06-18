from __future__ import annotations

import json
import shutil
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .dataset import DatasetRow
from .graph import DAGBuilder, dag_to_dict
from .state import ProofState


EXAMPLE_ID_WIDTH = 9


def example_stem(row_index: int) -> str:
    return f"{row_index:0{EXAMPLE_ID_WIDTH}d}"


def _reset_output_root(root: Path, *, force: bool) -> Path:
    if root.exists():
        if not force:
            raise FileExistsError(
                f"Output root '{root}' already exists. Re-run with --force to overwrite it."
            )
        if root.is_dir():
            shutil.rmtree(root)
        else:
            root.unlink()
    return root


def prepare_output_root(output_root: str | Path, *, splits: list[str], force: bool) -> Path:
    root = Path(output_root)
    _reset_output_root(root, force=force)

    for split in splits:
        (root / split / "json").mkdir(parents=True, exist_ok=True)
        (root / split / "pyg").mkdir(parents=True, exist_ok=True)

    for dirname in ("failures", "manifests", "reports", "vocab"):
        (root / dirname).mkdir(parents=True, exist_ok=True)
    for split in splits:
        (root / "failures" / f"{split}.jsonl").touch()

    return root


def prepare_audit_output_root(output_root: str | Path, *, splits: list[str], force: bool) -> Path:
    root = Path(output_root)
    _reset_output_root(root, force=force)

    for dirname in ("failures", "manifests", "reports"):
        (root / dirname).mkdir(parents=True, exist_ok=True)
    for split in splits:
        (root / "failures" / f"{split}.jsonl").touch()

    return root


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _relative_to(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def write_json_artifact(
    output_root: str | Path,
    *,
    split: str,
    row_index: int,
    payload: dict[str, object],
) -> Path:
    root = Path(output_root)
    path = root / split / "json" / f"{example_stem(row_index)}.json"
    return _write_json(path, payload)


def write_pyg_artifact(output_root: str | Path, *, split: str, row_index: int, data) -> Path:
    import torch

    root = Path(output_root)
    path = root / split / "pyg" / f"{example_stem(row_index)}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    return path


def append_failure_record(output_root: str | Path, *, split: str, record: dict[str, object]) -> Path:
    root = Path(output_root)
    path = root / "failures" / f"{split}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
    return path


def write_vocab(output_root: str | Path, *, name: str, vocab: dict[str, int]) -> Path:
    root = Path(output_root)
    path = root / "vocab" / name
    return _write_json(path, vocab)


def write_manifest(output_root: str | Path, *, split: str, manifest: dict[str, object]) -> Path:
    root = Path(output_root)
    path = root / "manifests" / f"{split}.json"
    return _write_json(path, manifest)


def write_summary_json(output_root: str | Path, summary: dict[str, object]) -> Path:
    root = Path(output_root)
    path = root / "reports" / "summary.json"
    return _write_json(path, summary)


def write_summary_markdown(output_root: str | Path, summary: dict[str, object]) -> Path:
    root = Path(output_root)
    path = root / "reports" / "summary.md"
    lines = [
        "# Prepared Dataset Summary",
        "",
        f"- dataset: `{summary['dataset']}`",
        f"- output root: `{summary['output_root']}`",
        f"- processed splits: `{', '.join(summary['splits'])}`",
        f"- node vocab size: `{summary['node_vocab_size']}`",
        f"- tactic vocab size: `{summary['tactic_vocab_size']}`",
        f"- attempted examples: `{summary['overall']['attempted_count']}`",
        f"- successful examples: `{summary['overall']['success_count']}`",
        f"- failed examples: `{summary['overall']['failure_count']}`",
        "",
        "## Split Metrics",
        "",
        "| Split | Attempted | Success | Failure | Success Rate | Mean Nodes | Median Nodes | Mean Edges | Median Edges |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for split in summary["splits"]:
        split_summary = summary["splits_summary"][split]
        lines.append(
            "| "
            f"{split} | "
            f"{split_summary['attempted_count']} | "
            f"{split_summary['success_count']} | "
            f"{split_summary['failure_count']} | "
            f"{split_summary['parser_success_rate']:.3f} | "
            f"{split_summary['graph_stats']['node_count']['mean']:.2f} | "
            f"{split_summary['graph_stats']['node_count']['median']:.2f} | "
            f"{split_summary['graph_stats']['edge_count']['mean']:.2f} | "
            f"{split_summary['graph_stats']['edge_count']['median']:.2f} |"
        )

    argument_summary = summary["overall"]["argument_label_summary"]
    lines.extend(
        [
            "",
            "## Clean Argument Labels",
            "",
            f"- examples with parsed arguments: `{argument_summary['examples_with_arguments']}`",
            f"- examples with clean trainable arguments: `{argument_summary['examples_with_clean_arguments']}`",
            f"- raw-expression examples skipped for argument loss: `{argument_summary['raw_expression_example_count']}`",
            f"- total parsed arguments: `{argument_summary['total_parsed_argument_count']}`",
            f"- trainable argument targets: `{argument_summary['trainable_argument_count']}`",
            f"- trainable local targets: `{argument_summary['trainable_local_argument_count']}`",
            f"- trainable lemma targets: `{argument_summary['trainable_lemma_argument_count']}`",
            f"- skipped argument tokens: `{argument_summary['skipped_argument_count']}`",
            f"- trainable argument coverage: `{argument_summary['trainable_argument_coverage']:.3f}`",
            "",
            "| Split | Parsed Args | Trainable | Local | Lemma | Skipped | Coverage |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for split in summary["splits"]:
        arg_split_summary = summary["splits_summary"][split]["argument_label_summary"]
        lines.append(
            "| "
            f"{split} | "
            f"{arg_split_summary['total_parsed_argument_count']} | "
            f"{arg_split_summary['trainable_argument_count']} | "
            f"{arg_split_summary['trainable_local_argument_count']} | "
            f"{arg_split_summary['trainable_lemma_argument_count']} | "
            f"{arg_split_summary['skipped_argument_count']} | "
            f"{arg_split_summary['trainable_argument_coverage']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Tactic Summary",
            "",
            f"- train tactic classes: `{summary['train_tactic_class_count']}`",
            "- top tactic names:",
        ]
    )
    for item in summary["top_tactic_names"]:
        lines.append(f"  - `{item['name']}`: {item['count']}")

    lines.extend(["", "## Top Failure Categories", ""])
    if summary["top_failure_categories"]:
        for item in summary["top_failure_categories"]:
            lines.append(f"- `{item['name']}`: {item['count']}")
    else:
        lines.append("- none")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_parser_audit_json(output_root: str | Path, summary: dict[str, object]) -> Path:
    root = Path(output_root)
    path = root / "reports" / "parser_audit.json"
    return _write_json(path, summary)


def write_parser_audit_markdown(output_root: str | Path, markdown: str) -> Path:
    root = Path(output_root)
    path = root / "reports" / "parser_audit.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    return path


def build_json_payload(
    row: DatasetRow,
    *,
    parsed_state: ProofState,
    dag: DAGBuilder,
    tactic_name: str,
) -> dict[str, object]:
    dag_payload = dag_to_dict(dag)
    return {
        "metadata": {
            "dataset": row.dataset_name,
            "split": row.split,
            "row_index": row.row_index,
            "theorem": row.theorem,
            "tactic_raw": row.tactic,
            "tactic_name": tactic_name,
        },
        "proof_state": parsed_state.as_dict(),
        "graph": {
            "stats": dag_payload["stats"],
            "nodes": dag_payload["nodes"],
            "edges": dag_payload["edges"],
        },
    }


def _unwrap_failure(exc: Exception, phase: str | None) -> tuple[str, Exception]:
    failure_phase = phase or getattr(exc, "phase", None) or "unknown"
    failure_exc = getattr(exc, "cause", exc)
    return failure_phase, failure_exc


def build_failure_record(
    row: DatasetRow,
    exc: Exception,
    *,
    phase: str | None = None,
) -> dict[str, object]:
    failure_phase, failure_exc = _unwrap_failure(exc, phase)
    error_type = failure_exc.__class__.__name__
    failure_category = f"{failure_phase}:{error_type}"
    return {
        "dataset": row.dataset_name,
        "split": row.split,
        "row_index": row.row_index,
        "theorem": row.theorem,
        "tactic_raw": row.tactic,
        "phase": failure_phase,
        "failure_category": failure_category,
        "error_type": error_type,
        "error_message": str(failure_exc),
        "state_preview": "" if row.state is None else str(row.state)[:200],
    }


def _stats_dict(values: list[int]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0}
    return {
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
    }


def _counter_summary(counter: Counter[str], *, limit: int = 10) -> list[dict[str, object]]:
    return [{"name": name, "count": count} for (name, count) in counter.most_common(limit)]


@dataclass
class SplitReport:
    split: str
    attempted_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    node_counts: list[int] = field(default_factory=list)
    edge_counts: list[int] = field(default_factory=list)
    reused_node_counts: list[int] = field(default_factory=list)
    tactic_counts: Counter[str] = field(default_factory=Counter)
    failure_categories: Counter[str] = field(default_factory=Counter)
    failure_phases: Counter[str] = field(default_factory=Counter)
    representative_failures: dict[str, list[dict[str, object]]] = field(default_factory=dict)
    examples_with_arguments: int = 0
    examples_with_clean_arguments: int = 0
    raw_expression_example_count: int = 0
    total_parsed_argument_count: int = 0
    trainable_argument_count: int = 0
    trainable_local_argument_count: int = 0
    trainable_lemma_argument_count: int = 0
    skipped_argument_count: int = 0
    argument_category_counts: Counter[str] = field(default_factory=Counter)

    def record_success(self, *, dag: DAGBuilder, tactic_name: str) -> None:
        self.attempted_count += 1
        self.success_count += 1
        self.node_counts.append(dag.num_nodes)
        self.edge_counts.append(dag.num_edges)
        self.reused_node_counts.append(len(dag.reused_nodes()))
        self.tactic_counts[tactic_name] += 1

    def record_argument_labels(
        self,
        *,
        total_arguments: int,
        trainable_local: int,
        trainable_lemma: int,
        skipped_arguments: int,
        has_raw_expression: bool,
        category_counts: Counter[str],
    ) -> None:
        if total_arguments > 0:
            self.examples_with_arguments += 1
        if trainable_local + trainable_lemma > 0:
            self.examples_with_clean_arguments += 1
        if has_raw_expression:
            self.raw_expression_example_count += 1

        self.total_parsed_argument_count += total_arguments
        self.trainable_local_argument_count += trainable_local
        self.trainable_lemma_argument_count += trainable_lemma
        self.trainable_argument_count += trainable_local + trainable_lemma
        self.skipped_argument_count += skipped_arguments
        self.argument_category_counts.update(category_counts)

    def record_failure(
        self,
        *,
        category: str,
        phase: str | None = None,
        example: dict[str, object] | None = None,
        max_examples_per_category: int | None = None,
    ) -> None:
        self.attempted_count += 1
        self.failure_count += 1
        self.failure_categories[category] += 1
        resolved_phase = phase or category.partition(":")[0] or "unknown"
        self.failure_phases[resolved_phase] += 1

        if example is not None and (max_examples_per_category is None or max_examples_per_category > 0):
            bucket = self.representative_failures.setdefault(category, [])
            if max_examples_per_category is None or len(bucket) < max_examples_per_category:
                bucket.append(example)

    def parser_success_rate(self) -> float:
        if self.attempted_count == 0:
            return 0.0
        return self.success_count / self.attempted_count

    def to_manifest(
        self,
        *,
        dataset_name: str,
        output_root: str | Path,
        vocab_source: str,
        sample_limit: int | None,
    ) -> dict[str, object]:
        root = Path(output_root)
        json_dir = root / self.split / "json"
        pyg_dir = root / self.split / "pyg"
        failure_log = root / "failures" / f"{self.split}.jsonl"
        manifest_path = root / "manifests" / f"{self.split}.json"

        return {
            "dataset": dataset_name,
            "split": self.split,
            "sample_limit": sample_limit,
            "attempted_count": self.attempted_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "parser_success_rate": self.parser_success_rate(),
            "vocab_source": vocab_source,
            "artifact_paths": {
                "json_dir": _relative_to(json_dir, root),
                "pyg_dir": _relative_to(pyg_dir, root),
                "failure_log": _relative_to(failure_log, root),
                "manifest": _relative_to(manifest_path, root),
                "node_vocab": _relative_to(root / "vocab" / "node_vocab.json", root),
                "tactic_vocab": _relative_to(root / "vocab" / "tactic_vocab.json", root),
            },
            "graph_stats": {
                "node_count": _stats_dict(self.node_counts),
                "edge_count": _stats_dict(self.edge_counts),
                "reused_node_count": _stats_dict(self.reused_node_counts),
            },
            "tactic_class_count": len(self.tactic_counts),
            "top_tactics": _counter_summary(self.tactic_counts),
            "top_failure_categories": _counter_summary(self.failure_categories),
            "top_failure_phases": _counter_summary(self.failure_phases),
            "argument_label_summary": {
                "examples_with_arguments": self.examples_with_arguments,
                "examples_with_clean_arguments": self.examples_with_clean_arguments,
                "raw_expression_example_count": self.raw_expression_example_count,
                "total_parsed_argument_count": self.total_parsed_argument_count,
                "trainable_argument_count": self.trainable_argument_count,
                "trainable_local_argument_count": self.trainable_local_argument_count,
                "trainable_lemma_argument_count": self.trainable_lemma_argument_count,
                "skipped_argument_count": self.skipped_argument_count,
                "trainable_argument_coverage": (
                    0.0
                    if self.total_parsed_argument_count == 0
                    else self.trainable_argument_count / self.total_parsed_argument_count
                ),
                "category_counts": dict(self.argument_category_counts),
            },
        }

    def to_audit_manifest(
        self,
        *,
        dataset_name: str,
        output_root: str | Path,
        sample_limit: int | None,
    ) -> dict[str, object]:
        root = Path(output_root)
        failure_log = root / "failures" / f"{self.split}.jsonl"
        manifest_path = root / "manifests" / f"{self.split}.json"

        return {
            "dataset": dataset_name,
            "split": self.split,
            "sample_limit": sample_limit,
            "attempted_count": self.attempted_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "parser_success_rate": self.parser_success_rate(),
            "artifact_paths": {
                "failure_log": _relative_to(failure_log, root),
                "manifest": _relative_to(manifest_path, root),
                "parser_audit_json": _relative_to(root / "reports" / "parser_audit.json", root),
                "parser_audit_markdown": _relative_to(root / "reports" / "parser_audit.md", root),
            },
            "graph_stats": {
                "node_count": _stats_dict(self.node_counts),
                "edge_count": _stats_dict(self.edge_counts),
                "reused_node_count": _stats_dict(self.reused_node_counts),
            },
            "top_failure_categories": _counter_summary(self.failure_categories),
            "top_failure_phases": _counter_summary(self.failure_phases),
            "representative_failure_examples": self.representative_failures,
        }


def build_summary(
    *,
    dataset_name: str,
    output_root: str | Path,
    splits: list[str],
    manifests: dict[str, dict[str, object]],
    split_reports: dict[str, SplitReport],
    node_vocab: dict[str, int],
    tactic_vocab: dict[str, int],
) -> dict[str, object]:
    overall_attempted = sum(report.attempted_count for report in split_reports.values())
    overall_success = sum(report.success_count for report in split_reports.values())
    overall_failure = sum(report.failure_count for report in split_reports.values())
    overall_failure_categories: Counter[str] = Counter()
    overall_tactic_counts: Counter[str] = Counter()
    overall_argument_categories: Counter[str] = Counter()
    overall_examples_with_arguments = 0
    overall_examples_with_clean_arguments = 0
    overall_raw_expression_examples = 0
    overall_total_arguments = 0
    overall_trainable_arguments = 0
    overall_trainable_local_arguments = 0
    overall_trainable_lemma_arguments = 0
    overall_skipped_arguments = 0
    for report in split_reports.values():
        overall_failure_categories.update(report.failure_categories)
        overall_tactic_counts.update(report.tactic_counts)
        overall_argument_categories.update(report.argument_category_counts)
        overall_examples_with_arguments += report.examples_with_arguments
        overall_examples_with_clean_arguments += report.examples_with_clean_arguments
        overall_raw_expression_examples += report.raw_expression_example_count
        overall_total_arguments += report.total_parsed_argument_count
        overall_trainable_arguments += report.trainable_argument_count
        overall_trainable_local_arguments += report.trainable_local_argument_count
        overall_trainable_lemma_arguments += report.trainable_lemma_argument_count
        overall_skipped_arguments += report.skipped_argument_count

    return {
        "dataset": dataset_name,
        "output_root": str(Path(output_root)),
        "splits": splits,
        "node_vocab_size": len(node_vocab),
        "tactic_vocab_size": len(tactic_vocab),
        "train_tactic_class_count": len(split_reports["train"].tactic_counts),
        "overall": {
            "attempted_count": overall_attempted,
            "success_count": overall_success,
            "failure_count": overall_failure,
            "argument_label_summary": {
                "examples_with_arguments": overall_examples_with_arguments,
                "examples_with_clean_arguments": overall_examples_with_clean_arguments,
                "raw_expression_example_count": overall_raw_expression_examples,
                "total_parsed_argument_count": overall_total_arguments,
                "trainable_argument_count": overall_trainable_arguments,
                "trainable_local_argument_count": overall_trainable_local_arguments,
                "trainable_lemma_argument_count": overall_trainable_lemma_arguments,
                "skipped_argument_count": overall_skipped_arguments,
                "trainable_argument_coverage": (
                    0.0
                    if overall_total_arguments == 0
                    else overall_trainable_arguments / overall_total_arguments
                ),
                "category_counts": dict(overall_argument_categories),
            },
        },
        "splits_summary": manifests,
        "top_tactic_names": _counter_summary(overall_tactic_counts),
        "top_failure_categories": _counter_summary(overall_failure_categories),
    }
