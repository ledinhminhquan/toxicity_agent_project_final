#!/usr/bin/env bash
set -euo pipefail

# Example script for Colab.
# Assumes you've mounted Drive and set ARTIFACTS_DIR, TOXICITY_MODEL_DIR, TOXICITY_RUN_DIR

toxicity-agent train --config configs/train.yaml
