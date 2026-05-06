#!/usr/bin/env bash
# run_seed_sweep_weather.sh — Run all 5 variants × 4 horizons × 3 seeds on Weather.
#
# Usage:
#   bash scripts/run_seed_sweep_weather.sh
#
# Downloads the Weather CSV first if not already present.
# Seeds: 42, 123, 456

set -euo pipefail

WEATHER_CSV="data/raw/weather/weather.csv"
if [ ! -f "$WEATHER_CSV" ]; then
  echo "Weather CSV not found. Downloading..."
  python scripts/download_weather.py
fi

CONFIGS=(
  "configs/baseline_weather.yaml"
  "configs/xcpe_weather.yaml"
  "configs/ablations/xcpe_early_weather.yaml"
  "configs/ablations/xcpe_late_weather.yaml"
  "configs/ablations/rope_weather.yaml"
)

HORIZONS=(96 192 336 720)
SEEDS=(42 123 456)

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
echo "All Weather seed sweeps done. Collecting results..."
python scripts/collect_results.py --dataset Weather
