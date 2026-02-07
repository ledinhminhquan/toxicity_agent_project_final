#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/train.yaml}
SPLIT=${2:-test}
THRESH=${3:-0.5}

toxicity-agent error-analysis --config "$CONFIG" --split "$SPLIT" --threshold "$THRESH"
