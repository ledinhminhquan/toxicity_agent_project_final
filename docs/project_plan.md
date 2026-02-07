# Project Plan / Timeline (Solo work, team-style)

Assume 2–3 weeks timeline (adjust to your course schedule).

## Phase 0 — Setup (Day 1)
- Create Git repo and folder structure
- Add configs, CI tests (optional)
- Verify Colab training pipeline works end-to-end on a small subset

Deliverable:
- Repo skeleton, `toxicity-agent demo-agent` works with Detoxify baseline.

## Phase 1 — Data (Days 2–3)
- Dataset selection + licensing check
- Preprocessing: normalization, split logic, negative downsampling option
- Data Description Document

Deliverable:
- `toxicity-agent eval` runs on baselines.

## Phase 2 — Baselines (Days 4–5)
- Baseline 1: TF-IDF + Logistic Regression
- Baseline 2: Detoxify pre-trained (unbiased/unbiased-small)
- Error analysis sample (false positives/negatives) *without printing toxic raw text*

Deliverable:
- Baseline metrics saved under `artifacts/runs/eval-*`.

## Phase 3 — Fine-tuning (Days 6–8)
- Fine-tune transformer model
- Basic hyperparameter tuning grid (learning rate, batch size, epochs)
- Compare vs baselines

Deliverable:
- Best model exported to `models/finetuned/latest`.

## Phase 4 — Agentic Moderation (Days 9–10)
- Implement routing logic (fast → second pass)
- PolicyStore rules + HumanReviewQueue
- Demonstrate example interactions

Deliverable:
- Agent architecture doc + demo.

## Phase 5 — Deployment (Days 11–12)
- FastAPI service
- Dockerfile
- API docs + examples

Deliverable:
- Running API with `/health` and `/v1/moderate`.

## Phase 6 — Monitoring & Continual Learning (Days 13–14)
- Logging design (privacy-preserving)
- Daily report job
- Continual learning strategy doc

Deliverable:
- Monitoring docs + sample report output.

## Phase 7 — Final report + slides (Days 15–18)
- 10–15 page report covering all rubric sections
- 10–15 slide deck aligned with report

Deliverable:
- PDF report + slides + repo link.
