from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .argument_labels import (
    GRAPH_NON_CANDIDATE,
    LIBRARY_LEMMA,
    LOCAL_HYPOTHESIS,
    RAW_EXPRESSION,
    UNRESOLVED,
    analyze_argument_labels,
)
from .dataset import DATASET_NAME, canonicalize_split_name, iter_dataset_rows
from .lemma_corpus import load_lemma_name_index
from .preparation import prepare_example
from .pyg import build_premise_mask
from .reporting import console_print


DEFAULT_ARGUMENT_AUDIT_OUTPUT_ROOT = Path("artifacts") / "audits" / "arguments" / "v1"


@dataclass(frozen=True)
class ArgumentAuditConfig:
    dataset_name: str = DATASET_NAME
    splits: tuple[str, ...] = ("train", "val", "test")
    output_root: Path = DEFAULT_ARGUMENT_AUDIT_OUTPUT_ROOT
    sample_per_split: int | None = None
    lemma_corpus_path: Path | None = None
    max_examples_per_category: int = 10
    force: bool = False


@dataclass
class SplitArgumentAudit:
    split: str
    attempted_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    examples_with_arguments: int = 0
    total_arguments: int = 0
    category_counts: Counter[str] = field(default_factory=Counter)
    tactic_counts: Counter[str] = field(default_factory=Counter)
    tactic_category_counts: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    representative_examples: dict[str, list[dict[str, object]]] = field(default_factory=lambda: defaultdict(list))

    def record_example(
        self,
        *,
        theorem: str | None,
        row_index: int,
        tactic_raw: str,
        tactic_name: str,
        arg_tokens: list[str],
        categories: list[str],
        max_examples_per_category: int,
    ) -> None:
        self.success_count += 1
        self.tactic_counts[tactic_name] += 1
        if arg_tokens:
            self.examples_with_arguments += 1
        self.total_arguments += len(arg_tokens)

        for category in categories:
            self.category_counts[category] += 1
            self.tactic_category_counts[tactic_name][category] += 1
            examples = self.representative_examples[category]
            if len(examples) < max_examples_per_category:
                examples.append(
                    {
                        "split": self.split,
                        "row_index": row_index,
                        "theorem": theorem,
                        "tactic_raw": tactic_raw,
                        "tactic_name": tactic_name,
                        "arg_tokens": arg_tokens,
                    }
                )

    def to_summary(self) -> dict[str, object]:
        resolution_count = (
            self.category_counts[LOCAL_HYPOTHESIS]
            + self.category_counts[LIBRARY_LEMMA]
        )
        unresolved_count = (
            self.category_counts[UNRESOLVED]
            + self.category_counts[GRAPH_NON_CANDIDATE]
            + self.category_counts[RAW_EXPRESSION]
        )
        return {
            "split": self.split,
            "attempted_count": self.attempted_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "examples_with_arguments": self.examples_with_arguments,
            "total_arguments": self.total_arguments,
            "resolved_argument_count": resolution_count,
            "unresolved_or_noisy_argument_count": unresolved_count,
            "argument_resolution_rate": (
                0.0 if self.total_arguments == 0 else resolution_count / self.total_arguments
            ),
            "category_counts": dict(self.category_counts),
            "top_tactics": [
                {"tactic": tactic, "count": count}
                for tactic, count in self.tactic_counts.most_common(20)
            ],
            "tactic_category_counts": {
                tactic: dict(counter)
                for tactic, counter in sorted(self.tactic_category_counts.items())
            },
            "representative_examples": dict(self.representative_examples),
        }


def _normalize_splits(raw_splits: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(raw_splits, str):
        candidates = [part.strip() for part in raw_splits.split(",")]
    else:
        candidates = [part.strip() for part in raw_splits]
    splits = [canonicalize_split_name(split) for split in candidates if split]
    if not splits:
        raise ValueError("At least one split must be provided.")
    return list(dict.fromkeys(splits))


def _prepare_output_root(path: Path, *, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(
                f"Output root '{path}' already exists. Re-run with --force to overwrite it."
            )
        shutil.rmtree(path)
    (path / "reports").mkdir(parents=True, exist_ok=True)


def _audit_split(
    *,
    config: ArgumentAuditConfig,
    split: str,
    lemma_name_index: dict[str, int] | None,
) -> SplitArgumentAudit:
    report = SplitArgumentAudit(split=split)
    for row in iter_dataset_rows(
        dataset_name=config.dataset_name,
        split=split,
        sample_limit=config.sample_per_split,
    ):
        report.attempted_count += 1
        try:
            example = prepare_example(row)
            premise_mask = build_premise_mask(example.dag)
            analysis = analyze_argument_labels(
                raw_tactic=example.row.tactic,
                dag=example.dag,
                lemma_name_index=lemma_name_index,
                premise_mask=premise_mask,
            )
        except Exception:
            report.failure_count += 1
            continue

        categories = [resolution.category for resolution in analysis.resolutions]
        if analysis.has_raw_expression_argument:
            categories.append(RAW_EXPRESSION)
        report.record_example(
            theorem=example.row.theorem,
            row_index=example.row.row_index,
            tactic_raw=example.row.tactic,
            tactic_name=analysis.tactic_name,
            arg_tokens=analysis.arg_tokens,
            categories=categories,
            max_examples_per_category=config.max_examples_per_category,
        )
    return report


def _build_summary(config: ArgumentAuditConfig, split_reports: dict[str, SplitArgumentAudit]) -> dict[str, object]:
    overall = SplitArgumentAudit(split="overall")
    for report in split_reports.values():
        overall.attempted_count += report.attempted_count
        overall.success_count += report.success_count
        overall.failure_count += report.failure_count
        overall.examples_with_arguments += report.examples_with_arguments
        overall.total_arguments += report.total_arguments
        overall.category_counts.update(report.category_counts)
        overall.tactic_counts.update(report.tactic_counts)
        for tactic, counter in report.tactic_category_counts.items():
            overall.tactic_category_counts[tactic].update(counter)
        for category, examples in report.representative_examples.items():
            overall.representative_examples[category].extend(
                examples[: config.max_examples_per_category]
            )
            overall.representative_examples[category] = overall.representative_examples[category][
                : config.max_examples_per_category
            ]

    return {
        "dataset": config.dataset_name,
        "output_root": str(config.output_root),
        "splits": list(config.splits),
        "sample_per_split": config.sample_per_split,
        "lemma_corpus_path": None if config.lemma_corpus_path is None else str(config.lemma_corpus_path),
        "overall": overall.to_summary(),
        "splits_summary": {
            split: report.to_summary()
            for split, report in split_reports.items()
        },
    }


def _render_markdown(summary: dict[str, object]) -> str:
    overall = summary["overall"]
    lines = [
        "# Argument Label Audit",
        "",
        f"- dataset: `{summary['dataset']}`",
        f"- output root: `{summary['output_root']}`",
        f"- processed splits: `{', '.join(summary['splits'])}`",
        f"- sample per split: `{summary['sample_per_split']}`",
        f"- lemma corpus: `{summary['lemma_corpus_path']}`",
        f"- attempted examples: `{overall['attempted_count']}`",
        f"- successful examples: `{overall['success_count']}`",
        f"- examples with arguments: `{overall['examples_with_arguments']}`",
        f"- total parsed arguments: `{overall['total_arguments']}`",
        f"- argument resolution rate: `{overall['argument_resolution_rate']:.3f}`",
        "",
        "## Category Counts",
        "",
        "| Category | Count |",
        "| --- | ---: |",
    ]
    for category, count in sorted(overall["category_counts"].items()):
        lines.append(f"| `{category}` | {count} |")

    lines.extend(
        [
            "",
            "## Split Metrics",
            "",
            "| Split | Attempted | Success | Args | Resolution Rate |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for split in summary["splits"]:
        split_summary = summary["splits_summary"][split]
        lines.append(
            f"| {split} | {split_summary['attempted_count']} | "
            f"{split_summary['success_count']} | {split_summary['total_arguments']} | "
            f"{split_summary['argument_resolution_rate']:.3f} |"
        )

    lines.extend(["", "## Tactic Family Breakdown", ""])
    tactic_counts = overall["tactic_category_counts"]
    if tactic_counts:
        lines.extend(["| Tactic | local | lemma | graph non-candidate | unresolved | raw expression |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
        for tactic, counter in sorted(tactic_counts.items()):
            lines.append(
                f"| `{tactic}` | "
                f"{counter.get(LOCAL_HYPOTHESIS, 0)} | "
                f"{counter.get(LIBRARY_LEMMA, 0)} | "
                f"{counter.get(GRAPH_NON_CANDIDATE, 0)} | "
                f"{counter.get(UNRESOLVED, 0)} | "
                f"{counter.get(RAW_EXPRESSION, 0)} |"
            )
    else:
        lines.append("- no parsed argument labels")

    lines.extend(["", "## Representative Noisy Examples", ""])
    noisy_categories = [RAW_EXPRESSION, GRAPH_NON_CANDIDATE, UNRESOLVED]
    examples_by_category = overall["representative_examples"]
    for category in noisy_categories:
        lines.append(f"### `{category}`")
        examples = examples_by_category.get(category, [])
        if not examples:
            lines.append("")
            lines.append("- none")
            lines.append("")
            continue
        lines.append("")
        for example in examples:
            lines.append(
                "- "
                f"split=`{example['split']}` row=`{example['row_index']}` "
                f"theorem=`{example['theorem'] or '<unknown>'}` "
                f"tactic=`{example['tactic_raw']}` "
                f"args=`{example['arg_tokens']}`"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def run_argument_label_audit(config: ArgumentAuditConfig) -> dict[str, object]:
    output_root = Path(config.output_root)
    _prepare_output_root(output_root, force=config.force)

    lemma_name_index = None
    if config.lemma_corpus_path is not None:
        lemma_name_index = load_lemma_name_index(config.lemma_corpus_path)

    split_reports: dict[str, SplitArgumentAudit] = {}
    for split in config.splits:
        console_print(f"\n  Auditing argument labels for split '{split}'...")
        split_reports[split] = _audit_split(
            config=config,
            split=split,
            lemma_name_index=lemma_name_index,
        )

    summary = _build_summary(config, split_reports)
    reports_dir = output_root / "reports"
    json_path = reports_dir / "argument_label_audit.json"
    md_path = reports_dir / "argument_label_audit.md"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(_render_markdown(summary), encoding="utf-8")

    console_print(f"\n  Wrote argument audit JSON    : {json_path}")
    console_print(f"  Wrote argument audit Markdown: {md_path}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit tactic argument-label quality")
    parser.add_argument("--dataset-name", type=str, default=DATASET_NAME)
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_ARGUMENT_AUDIT_OUTPUT_ROOT))
    parser.add_argument("--sample-per-split", type=int, default=None)
    parser.add_argument("--lemma-corpus", type=str, default=None)
    parser.add_argument("--max-examples-per-category", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        config = ArgumentAuditConfig(
            dataset_name=args.dataset_name,
            splits=tuple(_normalize_splits(args.splits)),
            output_root=Path(args.output_root),
            sample_per_split=args.sample_per_split,
            lemma_corpus_path=None if args.lemma_corpus is None else Path(args.lemma_corpus),
            max_examples_per_category=args.max_examples_per_category,
            force=args.force,
        )
        run_argument_label_audit(config)
    except (FileExistsError, RuntimeError, ValueError) as exc:
        console_print(f"  ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
