# Continual Learning & Monitoring Strategy

## What to monitor (production)
**Model quality signals**
- Distribution of predicted toxicity scores over time
- Fraction of content in each action bucket (ALLOW/WARN/BLOCK/REVIEW)
- If human labels exist: rolling F1/precision/recall

**Data drift signals**
- Input length distribution drift
- Language distribution drift
- OOV / tokenizer unknown rate proxy (subword length statistics)
- Topic drift (optional: embedding clustering)

## How new data is collected
- Collect samples from:
  - Human-reviewed borderline content (REVIEW queue)
  - User appeals / moderator overrides (false positives)
  - Newly emerging slang or adversarial attempts

Store only what is necessary:
- Prefer hashed identifiers + redacted text (or store raw text only in secure storage).

## Retraining / fine-tuning loop (conceptual)
1) Weekly/monthly export a curated dataset from review/feedback.
2) Label with clear guidelines + double annotation for reliability.
3) Re-train model with:
   - the previous training set (to avoid catastrophic forgetting),
   - plus new curated examples.
4) Evaluate against:
   - held-out test set
   - fairness slices
   - latency constraints
5) Deploy with model versioning and canary testing.
6) Monitor for regression; rollback if needed.

## Drift detection example
This repo includes a simple daily report job. In a real system you can add:
- statistical drift tests on embeddings (e.g., MMD)
- alerting via Slack/email
