from dataclasses import dataclass
from typing import Dict

from fastapi.testclient import TestClient

from toxicity_agent.api.main import app
from toxicity_agent.api import dependencies
from toxicity_agent.agent.actions import Action, ModerationDecision


class DummyAgent:
    def moderate(self, text: str) -> ModerationDecision:
        return ModerationDecision(
            action=Action.ALLOW,
            overall_score=0.1,
            label_scores={"toxic": 0.1},
            model_used="dummy",
            language="en",
            policy_message="ok",
            explanation="none",
            request_human_review=False,
        )


class DummyLogger:
    def log(self, text: str, payload: Dict, redacted_text=None) -> None:
        return


def test_api_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_api_moderate_single():
    app.dependency_overrides[dependencies.get_agent] = lambda: DummyAgent()
    app.dependency_overrides[dependencies.get_prediction_logger] = lambda: DummyLogger()
    client = TestClient(app)
    r = client.post("/v1/moderate", json={"text": "hello"})
    assert r.status_code == 200
    data = r.json()
    assert "results" in data
    assert len(data["results"]) == 1
    assert data["results"][0]["action"] == "ALLOW"
