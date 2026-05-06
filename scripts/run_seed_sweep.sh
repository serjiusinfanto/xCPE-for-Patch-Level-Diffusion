#!/usr/bin/env bash
# run_seed_sweep.sh — Run all 5 variants × 4 horizons × 2 extra seeds (123, 456).
#
# Usage:
#   bash scripts/run_seed_sweep.sh
#
# Seeds 123 and 456 are run here; seed 42 was already done manually.

set -euo pipefail

CONFIGS=(
  "configs/baseline_etth1.yaml"
  "configs/xcpe_etth1.yaml"
  "configs/ablations/xcpe_early_layers.yaml"
  "configs/ablations/xcpe_late_layers.yaml"
  "configs/ablations/rope_etth1.yaml"
)

HORIZONS=(96 192 336 720)
SEEDS=(123 456)

for SEED in "${SEEDS[@]}"; do
  for H in "${HORIZONS[@]}"; do
    for CFG in "${CONFIGS[@]}"; do
      echo ""
      echo "========================================================"
      echo "  config=$CFG  horizon=$H  seed=$SEED"
      echo "========================================================"

      bash scripts/run_pretrain.sh --config "$CFG" --horizon "$H" --seed "$SEED"
      bash scripts/run_finetune.sh --config "$CFG" --horizon "$H" --seed "$SEED"
    done
  done
done

echo ""
echo "All seed sweeps done. Collecting results..."
python scripts/collect_results.py --dataset ETTh1
