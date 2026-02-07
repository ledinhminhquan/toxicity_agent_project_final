from dataclasses import dataclass
from pathlib import Path

import numpy as np

from toxicity_agent.agent.moderation_agent import ModerationAgent
from toxicity_agent.agent.policy_store import PolicyStore
from toxicity_agent.agent.actions import Action
from toxicity_agent.agent.language import LanguageDetector
from toxicity_agent.models.model_registry import Predictors


class DummyMatrixPredictor:
    def __init__(self, mat):
        self._mat = np.array(mat, dtype=np.float32)
        self.model_type = "dummy"

    def predict_proba_matrix(self, texts, label_order, label_map=None):
        # Repeat the row for batch size
        row = self._mat.reshape(1, -1)
        return np.repeat(row, repeats=len(texts), axis=0)


class DummyFinetuned:
    def __init__(self, mat):
        self._mat = np.array(mat, dtype=np.float32)
        self.model_dir = Path("dummy_finetuned")

    def predict_proba_matrix(self, texts, label_order):
        row = self._mat.reshape(1, -1)
        return np.repeat(row, repeats=len(texts), axis=0)


class FixedLang(LanguageDetector):
    def __init__(self, lang):
        super().__init__(default=lang)
        self._lang = lang

    def detect(self, text: str) -> str:
        return self._lang


def _policy():
    # Minimal policy for tests
    rules = {
        "actions": {
            "ALLOW": {"user_message": "ok"},
            "WARN": {"user_message": "warn"},
            "REVIEW": {"user_message": "review"},
            "BLOCK": {"user_message": "block"},
        },
        "labels": {"toxic": {"display_name": "tox"}},
        "explanations": {"top_k_labels": 2, "include_scores": True},
    }
    return PolicyStore(rules=rules)


def test_agent_allow_low_risk():
    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    fast = DummyMatrixPredictor([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    multi = DummyMatrixPredictor([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])

    predictors = Predictors(
        label_fields=labels,
        label_map_detoxify={"identity_hate": "identity_attack"},
        finetuned=None,
        detoxify_fast=fast,
        detoxify_multilingual=multi,
    )

    agent = ModerationAgent(
        predictors=predictors,
        policy=_policy(),
        lang_detector=FixedLang("en"),
        review_queue=None,
        cfg={
            "agent": {
                "borderline_low": 0.35,
                "borderline_high": 0.65,
                "high_risk_threshold": 0.85,
                "action_thresholds": {"allow_lt": 0.3, "warn_lt": 0.6, "block_gte": 0.6},
                "prefer_multilingual_when_not_english": True,
            }
        },
    )

    d = agent.moderate("hello")
    assert d.action == Action.ALLOW


def test_agent_borderline_goes_to_review():
    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    fast = DummyMatrixPredictor([0.5, 0.1, 0.0, 0.0, 0.0, 0.0])  # borderline overall=0.5
    multi = DummyMatrixPredictor([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
    finetuned = DummyFinetuned([0.9, 0.2, 0.0, 0.0, 0.0, 0.0])

    predictors = Predictors(
        label_fields=labels,
        label_map_detoxify={"identity_hate": "identity_attack"},
        finetuned=finetuned,
        detoxify_fast=fast,
        detoxify_multilingual=multi,
    )

    agent = ModerationAgent(
        predictors=predictors,
        policy=_policy(),
        lang_detector=FixedLang("en"),
        review_queue=None,
        cfg={
            "agent": {
                "borderline_low": 0.35,
                "borderline_high": 0.65,
                "high_risk_threshold": 0.85,
                "action_thresholds": {"allow_lt": 0.3, "warn_lt": 0.6, "block_gte": 0.6},
                "prefer_multilingual_when_not_english": True,
            }
        },
    )

    d = agent.moderate("some text")
    assert d.action == Action.REVIEW
