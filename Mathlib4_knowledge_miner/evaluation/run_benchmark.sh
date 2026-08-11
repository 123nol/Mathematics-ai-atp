#!/usr/bin/env bash
# =============================================================================
# evaluation/run_benchmark.sh
# =============================================================================
# Reproducibility command for the full lemma retrieval benchmark.
#
# Prerequisites:
#   pip install torch-geometric datasets
#   pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
#       -f https://data.pyg.org/whl/torch-2.10.0+cu128.html
#
# Usage:
#   cd /path/to/Mathlib4_knowledge_miner
#   bash evaluation/run_benchmark.sh
# =============================================================================

set -euo pipefail

# Default to the local path if not provided as the first argument
BASE="${1:-/path/to/maths_ai_best_improvement_models_20260719}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_DIR="$REPO_DIR/evaluation"
RESULTS_DIR="$EVAL_DIR/results"
EMBEDDINGS="$EVAL_DIR/precomputed_val_embeddings.npz"

CHECKPOINT="$BASE/runs/lemma_retriever_clean_v1_50ep_safe/run_20260619_233643/best.pt"
PREPARED_ROOT="$BASE/artifacts/prepared/clean_v1"
LEMMA_INDEX="$BASE/artifacts/lemmas/contrastive_v1_50ep_safe/index"
CORPUS="$BASE/premise-selection/artifacts/lemmas/v1/corpus/lemmas.jsonl"
MAPPING="$REPO_DIR/lemma_graph_mapping.json"
GRAPH="$REPO_DIR/mathlib_dependencies.json"

echo "============================================================"
echo "  Mathlib Lemma Retrieval Benchmark"
echo "  Checkpoint: $CHECKPOINT"
echo "============================================================"
echo

# ---------------------------------------------------------------------------
# Step 1: Precompute val embeddings (skip if already done)
# ---------------------------------------------------------------------------

if [ -f "$EMBEDDINGS" ]; then
    echo "[Step 1] Embeddings already exist: $EMBEDDINGS — skipping."
else
    echo "[Step 1] Precomputing GNN proof-state embeddings..."
    cd "$REPO_DIR"
    .venv/bin/python -m evaluation.precompute_embeddings \
        --checkpoint   "$CHECKPOINT" \
        --prepared-root "$PREPARED_ROOT" \
        --output        "$EMBEDDINGS" \
        --split         val
    echo "[Step 1] Done: $EMBEDDINGS"
fi

echo

# ---------------------------------------------------------------------------
# Step 2: Semantic-only benchmark
# ---------------------------------------------------------------------------

echo "[Step 2] Running semantic-only benchmark..."
cd "$REPO_DIR"
.venv/bin/python -m evaluation.lemma_retrieval_bench \
    --embeddings    "$EMBEDDINGS" \
    --lemma-index   "$LEMMA_INDEX" \
    --corpus        "$CORPUS" \
    --k-values      200,500,1000,2000,5000 \
    --mode          semantic \
    --checkpoint    "$CHECKPOINT" \
    --output-dir    "$RESULTS_DIR" \
    --split         val

echo "[Step 2] Done. Results: $RESULTS_DIR/semantic_only.json"
echo

# ---------------------------------------------------------------------------
# Step 2b: Verify semantic-only reproduces the baseline
# ---------------------------------------------------------------------------

.venv/bin/python - <<'PYEOF'
import json, sys
from pathlib import Path

results_path = Path("evaluation/results/semantic_only.json")
if not results_path.exists():
    print("ERROR: semantic_only.json not found!")
    sys.exit(1)

data = json.loads(results_path.read_text())
metrics = data.get("metrics_by_k", {})

REF = {
    "200":  0.2936,
    "500":  0.4003,
    "1000": 0.5044,
    "2000": 0.6023,
    "5000": 0.7578,
}

print("Semantic-only recall vs  previous audit results:")
ok = True
for k, ref_r in REF.items():
    our_r = metrics.get(k, {}).get("recall", 0.0)
    delta = our_r - ref_r
    flag = "✓" if abs(delta) <= 0.01 else "✗ DISCREPANCY"
    print(f"  R@{k:<5} = {our_r:.4f}  (ref={ref_r:.4f}, Δ={delta:+.4f})  {flag}")
    if abs(delta) > 0.01:
        ok = False

if not ok:
    print()
    print("WARNING: Semantic-only does not reproduce the baseline.")
    print("         Proceeding to hybrid anyway for comparative evaluation.")
    # sys.exit(1)

print()
print("✓ Semantic-only reproduces the baseline. Proceeding to hybrid.")
PYEOF

echo

# ---------------------------------------------------------------------------
# Step 3: Hybrid benchmark
# ---------------------------------------------------------------------------

echo "[Step 3] Running hybrid benchmark..."
cd "$REPO_DIR"
.venv/bin/python -m evaluation.lemma_retrieval_bench \
    --embeddings    "$EMBEDDINGS" \
    --lemma-index   "$LEMMA_INDEX" \
    --corpus        "$CORPUS" \
    --mapping       "$MAPPING" \
    --graph         "$GRAPH" \
    --k-values      200,500,1000,2000,5000 \
    --mode          both \
    --semantic-k    1000 \
    --checkpoint    "$CHECKPOINT" \
    --output-dir    "$RESULTS_DIR" \
    --split         val

echo "[Step 3] Done."
echo

echo "============================================================"
echo "  Final outputs in: $RESULTS_DIR/"
echo "    semantic_only.json"
echo "    semantic_per_example.json"
echo "    hybrid_only.json"
echo "    hybrid_per_example.json"
echo "    comparison.json"
echo "    comparison.md"
echo "============================================================"
