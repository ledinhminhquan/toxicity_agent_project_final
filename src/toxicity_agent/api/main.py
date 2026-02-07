from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..logging_utils import setup_logging
from ..agent.moderation_agent import ModerationAgent
from ..monitoring.log_writer import PredictionLogger
from .dependencies import get_agent, get_prediction_logger
from .schemas import ModerateRequest, ModerateResponse, ModerationResult

setup_logging()

app = FastAPI(
    title="Toxicity Moderation Agent",
    version="0.1.0",
    description="Hate speech & toxicity detection with agentic routing and policy decisions",
)

# CORS (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/moderate", response_model=ModerateResponse)
def moderate(
    req: ModerateRequest,
    agent: ModerationAgent = Depends(get_agent),
    pred_logger: PredictionLogger = Depends(get_prediction_logger),
):
    texts = req.get_texts()
    results = []
    for text in texts:
        decision = agent.moderate(text)
        results.append(
            ModerationResult(
                action=decision.action.value,
                overall_score=decision.overall_score,
                label_scores=decision.label_scores,
                model_used=decision.model_used,
                language=decision.language,
                policy_message=decision.policy_message,
                explanation=decision.explanation,
                request_human_review=decision.request_human_review,
            )
        )
        pred_logger.log(
            text=text,
            redacted_text=decision.redacted_for_logs,
            payload={
                "action": decision.action.value,
                "overall_score": decision.overall_score,
                "label_scores": decision.label_scores,
                "model_used": decision.model_used,
                "language": decision.language,
                "request_human_review": decision.request_human_review,
            },
        )

    return ModerateResponse(results=results)
