#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/train.yaml}
FAIR_CFG=${2:-configs/fairness_slices.yaml}
SPLIT=${3:-test}
THRESH=${4:-0.5}

toxicity-agent fairness --config "$CONFIG" --fairness-config "$FAIR_CFG" --split "$SPLIT" --threshold "$THRESH"
