#!/usr/bin/env bash
set -euo pipefail

toxicity-agent serve --config configs/infer.yaml --host 0.0.0.0 --port 8000
