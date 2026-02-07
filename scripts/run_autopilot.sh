#!/usr/bin/env bash
set -euo pipefail

TRAIN_CONFIG=${1:-configs/train.yaml}
INFER_CONFIG=${2:-configs/infer.yaml}
FAIRNESS_CONFIG=${3:-configs/fairness_slices.yaml}

toxicity-agent autopilot \
  --train-config "$TRAIN_CONFIG" \
  --infer-config "$INFER_CONFIG" \
  --fairness-config "$FAIRNESS_CONFIG"
