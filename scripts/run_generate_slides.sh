#!/usr/bin/env bash
set -euo pipefail

TRAIN_CONFIG=${1:-configs/train.yaml}
INFER_CONFIG=${2:-configs/infer.yaml}

toxicity-agent generate-slides --train-config "$TRAIN_CONFIG" --infer-config "$INFER_CONFIG"
