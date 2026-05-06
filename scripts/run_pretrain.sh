#!/usr/bin/env bash
# Pre-training launcher.
#
# Usage:
#   bash scripts/run_pretrain.sh --config configs/baseline_etth1.yaml
#   bash scripts/run_pretrain.sh --config configs/baseline_etth1.yaml --horizon 192
#
# Runs via `accelerate launch` with FP16 mixed precision on a single GPU (RTX 4070).
# If VRAM is exhausted, edit the relevant YAML to reduce batch_size: 32 → 16.

set -euo pipefail

accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision fp16 \
  --dynamo_backend no \
  src/training/pretrain.py "$@"
