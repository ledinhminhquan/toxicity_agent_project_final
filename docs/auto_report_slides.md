# Auto-generated report & slide deck

This repository can auto-generate the two required submission artifacts:

- **Written report**: 10–15 pages (PDF)
- **Slides**: 10–15 slides (PPTX)

The generation uses JSON artifacts produced by the pipeline (eval, error analysis, fairness, benchmark, model metadata).
No raw toxic text is printed or stored in the generated artifacts.

## Commands

### End-to-end autopilot
```bash
toxicity-agent autopilot --train-config configs/train.yaml --infer-config configs/infer.yaml
```

### Generate report only
```bash
toxicity-agent generate-report --train-config configs/train.yaml --infer-config configs/infer.yaml
```

### Generate slides only
```bash
toxicity-agent generate-slides --train-config configs/train.yaml --infer-config configs/infer.yaml
```

Outputs are written under:
`artifacts/submission/submission-<timestamp>/`

## Notes

- The report is a **skeleton**: it includes the required sections and auto-fills numbers/figures when available.
- You should still review the narrative text and adjust it to match your exact use case.
- The slides are concise and visual; you can edit them in PowerPoint/Google Slides if needed.
