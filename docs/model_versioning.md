# Model Versioning (Metadata)

Each trained model folder contains `model_metadata.json` to support:
- reproducibility
- deployment traceability
- auditability

This aligns with the assignment requirement to consider **model versioning** for deployment.

## What's inside

- model name/type
- run id + timestamp
- git commit hash (if available)
- dataset name/fields/label list
- sample counts (train/val/test)
- training config subset
- metrics snapshot
- environment (python/torch/cuda/gpu name)

## Where it is written

After training:
- `artifacts/models/finetuned/<run-id>/model_metadata.json`
- also copied into `artifacts/models/finetuned/latest/`

No raw text or examples are stored.
