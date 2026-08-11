#!/usr/bin/env python3
"""
evaluation/lemma_retrieval_bench.py
====================================

Reproducible benchmark: Semantic-only vs Hybrid Mathlib lemma retrieval.

Implements the mentor's benchmark exactly:
    - Uses lemma_retriever_clean_v1_50ep_safe checkpoint for query embeddings
    - Queries FAISS IndexFlatIP with L2-normalized proof-state embeddings
    - Measures Recall@200/500/1000/2000/5000, MRR, median/mean hit rank
    - Denominator = proof steps with ≥1 library lemma argument (= 797 target samples)
    - Local-hypothesis-only tactics excluded from denominator
    - Multiple gold lemma IDs per tactic handled correctly

Expected baseline (must reproduce before proceeding to hybrid):
    R@200  = 0.2936  hit=234
    R@500  = 0.4003  hit=319
    R@1000 = 0.5044  hit=402
    R@2000 = 0.6023  hit=480
    R@5000 = 0.7578  hit=604
    MRR    ≈ 0.0401

Usage:
    # Step 1 (one-time): precompute proof-state embeddings
    python3 -m evaluation.precompute_embeddings \\
        --checkpoint "$BASE/runs/lemma_retriever_clean_v1_50ep_safe/run_20260619_233643/best.pt" \\
        --prepared-root "$BASE/artifacts/prepared/clean_v1" \\
        --output evaluation/precomputed_val_embeddings.npz

    # Step 2: semantic-only
    python3 -m evaluation.lemma_retrieval_bench \\
        --embeddings evaluation/precomputed_val_embeddings.npz \\
        --lemma-index "$BASE/artifacts/lemmas/contrastive_v1_50ep_safe/index" \\
        --corpus "$BASE/premise-selection/artifacts/lemmas/v1/corpus/lemmas.jsonl" \\
        --mode semantic \\
        --output-dir evaluation/results/

    # Step 3: hybrid
    python3 -m evaluation.lemma_retrieval_bench \\
        --embeddings evaluation/precomputed_val_embeddings.npz \\
        --lemma-index "$BASE/artifacts/lemmas/contrastive_v1_50ep_safe/index" \\
        --corpus "$BASE/premise-selection/artifacts/lemmas/v1/corpus/lemmas.jsonl" \\
        --mapping lemma_graph_mapping.json \\
        --graph mathlib_dependencies.json \\
        --mode both \\
        --output-dir evaluation/results/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.faiss_retriever import FAISSRetriever
from evaluation.gold_labels import (
    ArgCategory,
    ExclusionSummary,
    extract_gold_labels,
)
from evaluation.metrics import (
    RetrievalResult,
    compute_global_mrr,
    compute_metrics,
    find_first_hit_rank,
)


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_lemma_corpus(corpus_path: str | Path) -> dict[int, str]:
    """Load lemmas.jsonl → {lemma_id: name}."""
    corpus: dict[int, str] = {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            lid = int(obj["lemma_id"])
            corpus[lid] = str(obj["name"])
    return corpus


def build_lemma_name_index(corpus: dict[int, str]) -> dict[str, int]:
    """Invert corpus → {name: lemma_id}. First occurrence wins."""
    name_index: dict[str, int] = {}
    for lid, name in corpus.items():
        if name not in name_index:
            name_index[name] = lid
    return name_index


# ---------------------------------------------------------------------------
# Precomputed embeddings
# ---------------------------------------------------------------------------

def load_precomputed_embeddings(
    npz_path: str | Path,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Load precomputed embeddings .npz.

    Returns
    -------
    embeddings : (N, 512) float32
    tactics    : list[str], len N
    states     : list[str], len N
    """
    data = np.load(str(npz_path), allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32)
    tactics    = [str(t) for t in data.get("tactics", np.array([""] * len(embeddings)))]
    states     = [str(s) for s in data.get("states",  np.array([""] * len(embeddings)))]
    return embeddings, tactics, states


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    embeddings: np.ndarray,
    tactics: list[str],
    states: list[str],
    *,
    retriever,                          # FAISSRetriever or HybridReranker
    lemma_name_index: dict[str, int],
    k_values: list[int],
    mode: str = "semantic",
    semantic_k_for_hybrid: int = 1000,
    verbose: bool = True,
) -> tuple[list[RetrievalResult], ExclusionSummary]:
    """Run retrieval + evaluation for all proof states.

    Gold-label rules (matching mentor's pipeline):
        - library_lemma   → count as valid retrieval target
        - local_hypothesis → exclude from denominator
        - raw_expression   → exclude (unless also resolved to library lemma)
        - unresolved       → exclude
        - simp/ring/etc.   → no argument → exclude

    Multiple gold lemma IDs per tactic: all are tracked, first hit wins for MRR.
    """
    max_k = max(k_values)
    N     = len(embeddings)

    results: list[RetrievalResult] = []
    n_local   = 0
    n_raw     = 0
    n_unres   = 0
    n_no_lib  = 0
    n_has_lib = 0

    t0 = time.perf_counter()

    for i, (embedding, tactic, state) in enumerate(zip(embeddings, tactics, states)):
        if verbose and (i % 100 == 0 or i == N - 1):
            elapsed = time.perf_counter() - t0
            print(
                f"  [{i+1:>5}/{N}] elapsed={elapsed:>6.1f}s  "
                f"valid={n_has_lib}  excluded={i+1-n_has_lib}",
                end="\r",
                flush=True,
            )

        # --- Extract gold labels ---
        gold = extract_gold_labels(tactic, state, lemma_name_index=lemma_name_index)

        if not gold.has_library_lemma:
            categories = {a.category for a in gold.args}
            if not gold.args:
                reason = "no_library_lemma"
                n_no_lib += 1
            elif categories <= {ArgCategory.LOCAL_HYPOTHESIS}:
                reason = "local_hypothesis_only"
                n_local += 1
            elif ArgCategory.RAW_EXPRESSION in categories and ArgCategory.LIBRARY_LEMMA not in categories:
                reason = "raw_expression_only"
                n_raw += 1
            elif ArgCategory.UNRESOLVED in categories and ArgCategory.LIBRARY_LEMMA not in categories:
                reason = "unresolved_only"
                n_unres += 1
            else:
                reason = "no_library_lemma"
                n_no_lib += 1

            results.append(RetrievalResult(
                query_index=i,
                gold_lemma_ids=[],
                retrieved_lemma_ids=[],
                hit_rank=None,
                excluded=True,
                exclusion_reason=reason,
            ))
            continue

        n_has_lib += 1

        # --- Retrieve ---
        if mode == "semantic":
            retrieved = retriever.search(embedding, k=max_k)
        else:  # hybrid
            retrieved = retriever.hybrid_retrieve(
                embedding,
                semantic_k=semantic_k_for_hybrid,
                final_k=max_k,
            )

        # --- Find hit rank (multiple gold IDs supported) ---
        hit_rank = find_first_hit_rank(gold.library_lemma_ids, retrieved)

        results.append(RetrievalResult(
            query_index=i,
            gold_lemma_ids=gold.library_lemma_ids,
            retrieved_lemma_ids=retrieved[:200],  # store only top-200 for per-example output
            hit_rank=hit_rank,
        ))

    if verbose:
        print()

    exclusion = ExclusionSummary(
        total_processed=N,
        local_hypothesis_only=n_local,
        no_library_lemma_target=n_no_lib,
        unresolved_only=n_unres,
        raw_expression_only=n_raw,
        has_library_lemma=n_has_lib,
    )
    return results, exclusion


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_report(
    summary,
    exclusion: ExclusionSummary,
    *,
    mode: str,
    split: str = "val",
    checkpoint_path: str = "",
    index_path: str = "",
) -> None:
    SEP = "=" * 72

    print()
    print(SEP)
    print(f"  LEMMA RETRIEVAL BENCHMARK — {mode.upper()}")
    print(SEP)
    if checkpoint_path:
        print(f"  checkpoint : {checkpoint_path}")
    print(f"  index      : {index_path}")
    print(f"  split      : {split}")
    print()
    print("  EXCLUSION SUMMARY")
    print("-" * 72)
    print(exclusion)
    print()
    print("  METRICS")
    print("-" * 72)
    print(f"  target_samples (denominator) : {summary.target_samples}")
    print()
    print(summary.format_table())
    print()

    # Reference comparison for semantic mode
    if mode == "semantic":
        REF = {
            200: (234, 0.2936, 0.0395, 35.0,  56.35),
            500: (319, 0.4003, 0.0399, 73.0,  129.60),
            1000:(402, 0.5044, 0.0400, 133.0, 255.85),
            2000:(480, 0.6023, 0.0401, 211.5, 446.53),
            5000:(604, 0.7578, 0.0401, 432.0, 1036.95),
        }
        print("  COMPARISON TO PREVIOUS AUDIT (semantic)")
        print("-" * 72)
        hdr = f"  {'k':<6}  {'Recall':>8}  {'ref':>8}  {'Δ':>8}  {'MRR':>8}  {'ref':>8}  {'Δ':>8}"
        print(hdr)
        print("  " + "-" * 68)
        all_close = True
        for k in summary.k_values:
            ours_r = summary.recall.get(k, 0.0)
            ours_m = summary.mrr.get(k, 0.0)
            if k in REF:
                _, ref_r, ref_m, _, _ = REF[k]
                dr = ours_r - ref_r
                dm = ours_m - ref_m
                flag = ""
                if abs(dr) > 0.01:
                    flag = " ◄ DISCREPANCY"
                    all_close = False
                print(f"  {k:<6}  {ours_r:>8.4f}  {ref_r:>8.4f}  {dr:>+8.4f}  {ours_m:>8.4f}  {ref_m:>8.4f}  {dm:>+8.4f}{flag}")
        print()
        if all_close:
            print("  ✓ Semantic results are consistent with previous audit.")
        else:
            print("  ✗ DISCREPANCY DETECTED. Do NOT proceed to hybrid until this is resolved.")
        print()

    print(SEP)


def _make_per_example_output(results: list[RetrievalResult]) -> list[dict]:
    """Build per-example list for JSON output."""
    out = []
    for r in results:
        out.append({
            "query_index": r.query_index,
            "excluded": r.excluded,
            "exclusion_reason": r.exclusion_reason if r.excluded else None,
            "gold_lemma_ids": r.gold_lemma_ids,
            "hit_rank": r.hit_rank,
            "top_200_retrieved": r.retrieved_lemma_ids[:200],
        })
    return out


def _write_markdown_report(
    semantic_summary,
    hybrid_summary,
    *,
    output_path: Path,
) -> None:
    """Write comparison markdown report."""
    lines = [
        "# Lemma Retrieval Benchmark — Comparison Report\n",
        f"## Semantic vs Hybrid\n",
        "| k | Sem Recall | Hyb Recall | Δ Recall | Sem MRR | Hyb MRR | Δ MRR |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    k_vals = semantic_summary.k_values
    for k in k_vals:
        sr = semantic_summary.recall.get(k, 0.0)
        hr = hybrid_summary.recall.get(k, 0.0) if hybrid_summary else None
        sm = semantic_summary.mrr.get(k, 0.0)
        hm = hybrid_summary.mrr.get(k, 0.0) if hybrid_summary else None
        if hr is not None:
            dr = hr - sr
            dm = hm - sm
            lines.append(f"| {k} | {sr:.4f} | {hr:.4f} | {dr:+.4f} | {sm:.4f} | {hm:.4f} | {dm:+.4f} |")
        else:
            lines.append(f"| {k} | {sr:.4f} | — | — | {sm:.4f} | — | — |")

    lines.append("")
    lines.append("## Hit Rank Distribution\n")
    lines.append("| k | Sem Hits | Hyb Hits | Sem Median | Hyb Median | Sem Mean | Hyb Mean |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for k in k_vals:
        sh = semantic_summary.hit_counts.get(k, 0)
        hh = hybrid_summary.hit_counts.get(k, 0) if hybrid_summary else None
        sm = semantic_summary.median_hit_rank.get(k)
        hm = hybrid_summary.median_hit_rank.get(k) if hybrid_summary else None
        sa = semantic_summary.mean_hit_rank.get(k)
        ha = hybrid_summary.mean_hit_rank.get(k) if hybrid_summary else None
        sm_s  = f"{sm:.1f}"  if sm  is not None else "—"
        hm_s  = f"{hm:.1f}"  if hm  is not None else "—"
        sa_s  = f"{sa:.2f}"  if sa  is not None else "—"
        ha_s  = f"{ha:.2f}"  if ha  is not None else "—"
        hh_s  = str(hh)      if hh  is not None else "—"
        lines.append(f"| {k} | {sh} | {hh_s} | {sm_s} | {hm_s} | {sa_s} | {ha_s} |")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown report: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Benchmark semantic-only and hybrid Mathlib lemma retrieval.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--embeddings",
        required=True,
        help="Path to precomputed .npz (from precompute_embeddings.py).",
    )
    p.add_argument(
        "--lemma-index",
        required=True,
        help="Directory with faiss.index, lemma_ids.json, lemma_vectors.npy.",
    )
    p.add_argument(
        "--corpus",
        required=True,
        help="Path to lemmas.jsonl.",
    )
    p.add_argument("--mapping", default=None, help="lemma_graph_mapping.json (hybrid only).")
    p.add_argument("--graph",   default=None, help="mathlib_dependencies.json (hybrid only).")
    p.add_argument(
        "--k-values",
        default="200,500,1000,2000,5000",
        help="Comma-separated k values.",
    )
    p.add_argument(
        "--mode",
        choices=["semantic", "hybrid", "both"],
        default="semantic",
        help="Retrieval mode.",
    )
    p.add_argument(
        "--semantic-k",
        type=int,
        default=1000,
        help="FAISS candidates for hybrid mode graph seed.",
    )
    p.add_argument("--semantic-weight",  type=float, default=1.0)
    p.add_argument("--logical-weight",   type=float, default=0.75)
    p.add_argument("--frequency-weight", type=float, default=0.50)
    p.add_argument("--hop-weight",       type=float, default=0.25)
    p.add_argument("--max-hops",         type=int,   default=2)
    p.add_argument(
        "--output-dir",
        default="evaluation/results",
        help="Directory to write JSON/markdown outputs.",
    )
    p.add_argument("--split",      default="val",  help="Split name for reporting.")
    p.add_argument("--checkpoint", default="",     help="Checkpoint path (for reporting).")
    p.add_argument("--quiet",      action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args   = parser.parse_args(argv)

    k_values = [int(k.strip()) for k in args.k_values.split(",")]
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load ---
    print(f"\nLoading precomputed embeddings: {args.embeddings}")
    embeddings, tactics, states = load_precomputed_embeddings(args.embeddings)
    print(f"  {len(embeddings)} proof states × {embeddings.shape[1]}-dim")

    # Verify normalization (should be ~1.0 if precomputed correctly)
    norms = np.linalg.norm(embeddings, axis=1)
    if norms.max() > 1.1 or norms.min() < 0.9:
        print(
            f"\n  WARNING: embeddings do not appear to be L2-normalized.\n"
            f"  norm range: [{norms.min():.3f}, {norms.max():.3f}]\n"
            f"  Expected ~1.0. Results will not reproduce the previous audit.\n"
        )

    print(f"\nLoading lemma corpus: {args.corpus}")
    corpus = load_lemma_corpus(args.corpus)
    lemma_name_index = build_lemma_name_index(corpus)
    print(f"  {len(corpus):,} lemmas, {len(lemma_name_index):,} unique names")

    print(f"\nLoading FAISS index: {args.lemma_index}")
    faiss_retriever = FAISSRetriever.load(args.lemma_index)
    print(f"  {faiss_retriever._num_lemmas:,} lemmas, dim={faiss_retriever.index.d}")

    modes = ["semantic", "hybrid"] if args.mode == "both" else [args.mode]
    summaries: dict[str, object] = {}

    for mode in modes:
        print(f"\n{'='*72}")
        print(f"  Mode: {mode.upper()}")
        print(f"{'='*72}")

        if mode == "hybrid":
            if not args.mapping or not args.graph:
                print("ERROR: --mapping and --graph required for hybrid mode.")
                return 1
            from evaluation.hybrid_reranker import HybridReranker
            retriever = HybridReranker.load(
                index_dir=args.lemma_index,
                mapping_path=args.mapping,
                graph_path=args.graph,
                semantic_weight=args.semantic_weight,
                logical_weight=args.logical_weight,
                frequency_weight=args.frequency_weight,
                hop_weight=args.hop_weight,
                max_hops=args.max_hops,
            )
        else:
            retriever = faiss_retriever

        print(f"\nEvaluating {len(embeddings)} proof states...")
        results, exclusion = run_evaluation(
            embeddings, tactics, states,
            retriever=retriever,
            lemma_name_index=lemma_name_index,
            k_values=k_values,
            mode=mode,
            semantic_k_for_hybrid=args.semantic_k,
            verbose=not args.quiet,
        )

        summary = compute_metrics(results, k_values=k_values)
        summaries[mode] = summary

        print_report(
            summary, exclusion,
            mode=mode,
            split=args.split,
            checkpoint_path=args.checkpoint,
            index_path=str(args.lemma_index),
        )

        # Per-example results
        per_example = _make_per_example_output(results)
        per_path = out_dir / f"{mode}_per_example.json"
        per_path.write_text(
            json.dumps(per_example, indent=None, ensure_ascii=False),
            encoding="utf-8",
        )

        # Summary JSON
        summary_dict = {
            "mode": mode,
            "split": args.split,
            "checkpoint_path": args.checkpoint,
            "index_path": str(args.lemma_index),
            "k_values": k_values,
            **summary.to_dict(),
        }
        summary_path = out_dir / f"{mode}_only.json"
        summary_path.write_text(
            json.dumps(summary_dict, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  Saved: {summary_path}")
        print(f"  Saved: {per_path}")

    # Comparison
    if "semantic" in summaries and "hybrid" in summaries:
        sem_s = summaries["semantic"]
        hyb_s = summaries["hybrid"]

        comp = {
            "k_values": k_values,
            "comparison": {
                str(k): {
                    "semantic_recall": sem_s.recall.get(k, 0.0),
                    "hybrid_recall":   hyb_s.recall.get(k, 0.0),
                    "delta_recall":    hyb_s.recall.get(k, 0.0) - sem_s.recall.get(k, 0.0),
                    "semantic_mrr":    sem_s.mrr.get(k, 0.0),
                    "hybrid_mrr":      hyb_s.mrr.get(k, 0.0),
                    "delta_mrr":       hyb_s.mrr.get(k, 0.0) - sem_s.mrr.get(k, 0.0),
                    "semantic_hit_count": sem_s.hit_counts.get(k, 0),
                    "hybrid_hit_count":   hyb_s.hit_counts.get(k, 0),
                    "semantic_median_hit_rank": sem_s.median_hit_rank.get(k),
                    "hybrid_median_hit_rank":   hyb_s.median_hit_rank.get(k),
                    "semantic_mean_hit_rank": sem_s.mean_hit_rank.get(k),
                    "hybrid_mean_hit_rank":   hyb_s.mean_hit_rank.get(k),
                }
                for k in k_values
            },
        }
        comp_path = out_dir / "comparison.json"
        comp_path.write_text(json.dumps(comp, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n  Saved: {comp_path}")

        md_path = out_dir / "comparison.md"
        _write_markdown_report(sem_s, hyb_s, output_path=md_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
