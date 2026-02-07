# Data

This repository does **not** commit raw toxicity datasets (they contain sensitive/offensive text).
Instead, use the download/preprocessing scripts and store artifacts under your own `ARTIFACTS_DIR`.

Recommended:

- Raw cached datasets: `<ARTIFACTS_DIR>/data/raw/`
- Processed splits: `<ARTIFACTS_DIR>/data/processed/`

See `src/toxicity_agent/data/dataset.py` and `toxicity-agent train --config configs/train.yaml`.
