#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/infer.yaml}
N=${2:-300}
WARMUP=${3:-10}

toxicity-agent benchmark --config "$CONFIG" --n "$N" --warmup "$WARMUP"
