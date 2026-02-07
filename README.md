# Toxicity Moderation Agent (Detoxify + Fine-tuned Transformer)

An end-to-end **Hate Speech & Toxicity Detection** system designed like a production-ready NLP project:
- data handling
- baselines + fine-tuning
- agentic moderation (routing + policy + human review)
- deployable REST API (FastAPI)
- monitoring + continual learning plan

> ⚠️ Safety note: toxicity datasets contain offensive language. Do not print raw samples.

## Repository structure

```
project-root/
├── src/                  # Core source code (Python package)
├── data/                 # Data scripts (no raw data committed)
├── models/               # Trained models/checkpoints (not committed)
├── configs/              # YAML configs
├── tests/                # Unit tests
├── docs/                 # Assignment documents (Problem/Data/Ethics/etc.)
├── requirements.txt
└── README.md
```

This layout follows the required structure in the assignment brief.

## Quickstart (local)

### 1) Create env + install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2) Train the fine-tuned model
```bash
toxicity-agent train --config configs/train.yaml
```

### 3) Evaluate baselines and the fine-tuned model
```bash
toxicity-agent eval --config configs/train.yaml
```

### 4) Run the API
```bash
toxicity-agent serve --config configs/infer.yaml --host 0.0.0.0 --port 8000
```

Then open:
- `GET /health`
- `POST /v1/moderate`

Example:
```bash
curl -X POST "http://localhost:8000/v1/moderate" \
  -H "Content-Type: application/json" \
  -d '{"text":"You are being very rude today."}'
```

### 5) Run the agent demo (CLI)
```bash
toxicity-agent demo-agent --config configs/infer.yaml
```

## Training on Google Colab (recommended)

### Drive folder layout (recommended)

Create this structure in **Google Drive**:

```
MyDrive/
  NLP_Project/
    toxicity_agent/
      repo/                 # optional (if you want repo persisted)
      artifacts/
        data/
          raw/
          processed/
        models/
          finetuned/
          baselines/
        runs/
          benchmarks/
          error_analysis/
          fairness/
        review_queue/
```

### Colab steps (fast + reliable)

1) Mount Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

2) Clone repo into local disk (faster than Drive)
```bash
!git clone <YOUR_GIT_REPO_URL> /content/toxicity_agent
%cd /content/toxicity_agent
!pip install -r requirements.txt
!pip install -e .
```

3) Set env vars so artifacts go to Drive (persistent)
```bash
%env ARTIFACTS_DIR=/content/drive/MyDrive/NLP_Project/toxicity_agent/artifacts
%env TOXICITY_MODEL_DIR=/content/drive/MyDrive/NLP_Project/toxicity_agent/artifacts/models
%env TOXICITY_RUN_DIR=/content/drive/MyDrive/NLP_Project/toxicity_agent/artifacts/runs
```

4) Train
```bash
!toxicity-agent train --config configs/train.yaml
```

5) Serve (optional)
```bash
!toxicity-agent serve --config configs/infer.yaml --host 0.0.0.0 --port 8000
```

## Configuration

- `configs/train.yaml`: dataset + training hyperparameters
- `configs/infer.yaml`: inference + agent thresholds + logging
- `configs/policy_rules.yaml`: action messages and label names



## Enterprise/production-oriented extras (for higher score)

These help cover deployment + monitoring requirements:

### 1) Latency benchmark (p50/p95/p99)
```bash
toxicity-agent benchmark --config configs/infer.yaml --n 300 --warmup 10
```
Default output:
- `artifacts/runs/benchmarks/benchmark-<timestamp>.json`

### 2) Privacy-preserving error analysis (no raw toxic text stored)
```bash
toxicity-agent error-analysis --config configs/train.yaml --split test --threshold 0.5
```
Default output:
- `artifacts/runs/error_analysis/error-analysis-<timestamp>.json`

### 3) Fairness slice evaluation (identity-mention heuristic slices)
```bash
toxicity-agent fairness --config configs/train.yaml --fairness-config configs/fairness_slices.yaml --split test
```
Default output:
- `artifacts/runs/fairness/fairness-<timestamp>.json`

### 4) Model versioning metadata
After training, each saved model directory contains:
- `model_metadata.json` (training config subset, dataset info, metrics snapshot, env versions, git commit if available)


## Notes on baselines

- Baseline A: **Detoxify** pre-trained models (fast and strong)
- Baseline B: TF-IDF + Logistic Regression (simple)
- Main model: fine-tuned Transformer (HuggingFace)

## Tests
```bash
pytest -q
```

## License
MIT (for project code). Dataset licenses follow their respective sources.


## One-button autopilot (generate report + slides)

This repo includes an **autopilot pipeline** that runs:

- (optional) tuning
- training
- evaluation (baselines + fine-tuned)
- threshold search (validation split)
- latency benchmark (agent end-to-end)
- privacy-preserving error analysis
- fairness slice evaluation (identity mentions)
- **auto-generate**:
  - a **PDF report skeleton** (10–15 pages)
  - a **PPTX slide deck** (10–15 slides)

> Outputs are written under `artifacts/submission/submission-<timestamp>/`

### Run (local)
```bash
toxicity-agent autopilot \
  --train-config configs/train.yaml \
  --infer-config configs/infer.yaml \
  --fairness-config configs/fairness_slices.yaml \
  --title "Hate Speech & Toxicity Detection System" \
  --author "Your Name"
```


### Grading checklist (rubric completeness)

After running **autopilot**, the folder `artifacts/submission/submission-<timestamp>/` contains:

- `submission_manifest.json` (includes `grading_checklist` with PASS/WARN/FAIL per rubric item)
- `grading_checklist.json` (same content, separate file)
- `submission_bundle.zip` (report + slides + manifest + snapshot)

This is an automated *completeness* check (not a replacement for instructor grading).

### Generate report/slides only (after running train/eval)
```bash
toxicity-agent generate-report --train-config configs/train.yaml --infer-config configs/infer.yaml
toxicity-agent generate-slides --train-config configs/train.yaml --infer-config configs/infer.yaml
```

### Threshold calibration
```bash
toxicity-agent threshold-search --config configs/train.yaml --split validation
```
