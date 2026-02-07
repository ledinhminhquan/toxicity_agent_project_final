# System Architecture (Agentic Moderation)

## High-level components
1. **Inference models**
   - Baseline: Detoxify pre-trained models
   - Custom: fine-tuned Transformer (HuggingFace)

2. **Agentic moderation layer**
   - Language detection tool
   - Policy rule tool (thresholds, user messages)
   - Decision logic (route + action)
   - Human review queue tool (writes JSONL)

3. **Serving**
   - FastAPI REST service

4. **Monitoring & continual learning**
   - Privacy-preserving JSONL logs (hashes, scores, actions)
   - Daily report job (aggregates metrics)
   - Conceptual retraining loop

## Agent decision flow (pseudo)
Input: text
1) Detect language
2) Run fast model (Detoxify unbiased-small)
3) If borderline or high-risk:
      - if English and finetuned model available => run finetuned model
      - else => run Detoxify multilingual
4) Compute overall risk score
5) Consult policy thresholds => action (ALLOW/WARN/BLOCK/REVIEW)
6) If REVIEW => enqueue for human review
7) Return response: scores, action, explanation
