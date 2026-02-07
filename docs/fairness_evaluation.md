# Fairness Slice Evaluation (Identity Mentions)

This project includes a **heuristic** fairness evaluation based on *identity-term mentions*.
It is designed to be:
- **privacy-preserving** (no raw text is written)
- **reproducible** (JSON report saved under `artifacts/runs/fairness/`)

## Why this exists

The assignment requires ethical responsibility, bias/fairness risk discussion, and deployment considerations.

For toxicity detection, models can behave differently on texts that mention protected identities, even when the text is not hateful.

## What we measure

For each slice (e.g., `gender`, `religion`, ...), we compute:
- sample count
- label prevalence
- overall metrics (F1 micro/macro, ROC-AUC macro)
- per-label TPR/FPR
- gap metrics vs. a reference slice (`no_identity_mention`)

## Important limitations

- **Keyword matching is not demographic inference.**  
  A text mentioning an identity term does not imply the author or target belongs to that group.
- This is **not** a complete fairness audit.  
  Real deployments should combine:
  - more robust taxonomy
  - human review
  - domain-specific stress tests
  - continual monitoring

## How to run

```bash
toxicity-agent fairness --config configs/train.yaml --fairness-config configs/fairness_slices.yaml --split test
```
