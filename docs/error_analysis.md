# Error Analysis (Privacy-Preserving)

The assignment expects an error analysis section explaining what the model fails on and why.

To avoid printing or storing offensive content, we implement error analysis that:
- does **not** write raw text
- only outputs aggregate statistics + hashed identifiers (`sha256(text)`)

## What we compute

- overall metrics at a threshold
- confusion counts per label (TP/FP/FN/TN)
- feature summaries for FP vs FN:
  - length (chars/words)
  - uppercase ratio
  - punctuation ratio
  - URL/email presence
  - repeated characters
  - non-ascii ratio
- top error cases (hashed only) with:
  - per-label probabilities and true/pred labels
  - numeric features

## How to run

```bash
toxicity-agent error-analysis --config configs/train.yaml --split test --threshold 0.5
```
Output: `artifacts/runs/error_analysis/error-analysis-<timestamp>.json`
