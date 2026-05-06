#!/usr/bin/env bash
# Fine-tuning launcher.
#
# Usage:
#   bash scripts/run_finetune.sh --config configs/baseline_etth1.yaml
#   bash scripts/run_finetune.sh --config configs/baseline_etth1.yaml --horizon 720
#
# Requires a pre-trained checkpoint at:
#   results/checkpoints/<dataset>_h<horizon>_<variant>_pretrain_best.pt

set -euo pipefail

accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision fp16 \
  --dynamo_backend no \
  src/training/finetune.py "$@"
