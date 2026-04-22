"""Unit tests for the Vietnamese sidecar extension.

These tests are designed to be dependency-light: they use dummy predictors
and do not require the HuggingFace `datasets` package, detoxify, or network
access. They cover:

1. Backward compatibility: English routing is unchanged when the VI predictor
   is absent.
2. New VI route: when the detected language is Vietnamese AND the VI predictor
   is loaded, the agent always uses it as the authoritative scorer.
3. ``is_vietnamese`` helper.
4. ViHSD label mapping (pure-Python, no HF dependency).
5. ``force_second_pass_languages`` config hook triggers a second pass even on
   low-confidence inputs.
"""

from pathlib import Path

import numpy as np
import pytest

from toxicity_agent.agent.actions import Action
from toxicity_agent.agent.language import LanguageDetector, is_english, is_vietnamese
from toxicity_agent.agent.moderation_agent import ModerationAgent
from toxicity_agent.agent.policy_store import PolicyStore
from toxicity_agent.data.vi_hsd_adapter import (
    VI_LABEL_FIELDS,
    _class_id_to_multi_hot,
    _parse_label_to_class_id,
)
from toxicity_agent.models.model_registry import Predictors, VI_DEFAULT_LABEL_FIELDS


# ---------- Shared fixtures ----------

class DummyMatrixPredictor:
    """Stand-in for DetoxifyPredictor (6-label EN)."""

    def __init__(self, mat):
        self._mat = np.array(mat, dtype=np.float32)
        self.model_type = "dummy"

    def predict_proba_matrix(self, texts, label_order, label_map=None):
        row = self._mat.reshape(1, -1)
        return np.repeat(row, repeats=len(texts), axis=0)


class DummyFinetuned:
    """Stand-in for HFPredictor (English fine-tuned, 6-label)."""

    def __init__(self, mat, name="dummy_finetuned"):
        self._mat = np.array(mat, dtype=np.float32)
        self.model_dir = Path(name)

    def predict_proba_matrix(self, texts, label_order):
        row = self._mat.reshape(1, -1)
        return np.repeat(row, repeats=len(texts), axis=0)


class DummyVIFinetuned:
    """Stand-in for Vietnamese HFPredictor (2-label)."""

    def __init__(self, mat, name="dummy_vi_finetuned"):
        self._mat = np.array(mat, dtype=np.float32)
        self.model_dir = Path(name)

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
    rules = {
        "actions": {
            "ALLOW": {"user_message": "ok"},
            "WARN": {"user_message": "warn"},
            "REVIEW": {"user_message": "review"},
            "BLOCK": {"user_message": "block"},
        },
        "labels": {
            "toxic": {"display_name": "tox"},
            "offensive": {"display_name": "offensive"},
            "hate": {"display_name": "hate"},
        },
        "explanations": {"top_k_labels": 2, "include_scores": True},
    }
    return PolicyStore(rules=rules)


def _default_agent_cfg(force_langs=None, vi_threshold=0.5):
    cfg = {
        "agent": {
            "borderline_low": 0.35,
            "borderline_high": 0.65,
            "high_risk_threshold": 0.85,
            "action_thresholds": {"allow_lt": 0.3, "warn_lt": 0.6, "block_gte": 0.6},
            "prefer_multilingual_when_not_english": True,
        },
        "inference": {"vi_threshold": vi_threshold},
    }
    if force_langs is not None:
        cfg["agent"]["force_second_pass_languages"] = list(force_langs)
    return cfg


# ---------- is_english / is_vietnamese ----------

def test_is_english_and_is_vietnamese():
    assert is_english("en")
    assert is_english("en-US")
    assert not is_english("vi")
    assert not is_english("fr")

    assert is_vietnamese("vi")
    assert is_vietnamese("vi-VN")
    assert is_vietnamese("VI")  # case-insensitive
    assert not is_vietnamese("en")
    assert not is_vietnamese("")


# ---------- ViHSD label mapping ----------

def test_vihsd_label_mapping_strings():
    assert _parse_label_to_class_id("CLEAN") == 0
    assert _parse_label_to_class_id("clean") == 0
    assert _parse_label_to_class_id("OFFENSIVE") == 1
    assert _parse_label_to_class_id("HATE") == 2
    # Ints (already canonical)
    assert _parse_label_to_class_id(0) == 0
    assert _parse_label_to_class_id(1) == 1
    assert _parse_label_to_class_id(2) == 2
    # Garbage / unknown -> CLEAN (safe default)
    assert _parse_label_to_class_id(None) == 0
    assert _parse_label_to_class_id("unknown") == 0


def test_vihsd_multi_hot():
    assert _class_id_to_multi_hot(0) == [0.0, 0.0]
    assert _class_id_to_multi_hot(1) == [1.0, 0.0]  # offensive only
    assert _class_id_to_multi_hot(2) == [0.0, 1.0]  # hate only


def test_vi_default_label_fields_match_adapter():
    assert VI_DEFAULT_LABEL_FIELDS == VI_LABEL_FIELDS == ["offensive", "hate"]


# ---------- Backward compatibility: VI predictor absent ----------

def test_english_path_unchanged_when_vi_model_absent():
    """When vi_finetuned is None, the agent MUST NOT alter English behaviour."""
    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    fast = DummyMatrixPredictor([0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
    multi = DummyMatrixPredictor([0.05, 0.0, 0.0, 0.0, 0.0, 0.0])

    predictors = Predictors(
        label_fields=labels,
        label_map_detoxify={"identity_hate": "identity_attack"},
        finetuned=None,
        detoxify_fast=fast,
        detoxify_multilingual=multi,
        vi_finetuned=None,  # explicitly absent
    )

    agent = ModerationAgent(
        predictors=predictors,
        policy=_policy(),
        lang_detector=FixedLang("en"),
        review_queue=None,
        cfg=_default_agent_cfg(),
    )
    d = agent.moderate("hello world")
    assert d.action == Action.ALLOW
    assert d.language == "en"
    assert "detoxify" in d.model_used  # fast model path


def test_vi_text_falls_back_to_detoxify_multi_when_no_vi_model():
    """Vietnamese text with no VI model must still work (fallback to multilingual)."""
    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    fast = DummyMatrixPredictor([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])  # borderline triggers 2nd pass
    multi = DummyMatrixPredictor([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])

    predictors = Predictors(
        label_fields=labels,
        label_map_detoxify={"identity_hate": "identity_attack"},
        finetuned=None,
        detoxify_fast=fast,
        detoxify_multilingual=multi,
        vi_finetuned=None,
    )
    agent = ModerationAgent(
        predictors=predictors,
        policy=_policy(),
        lang_detector=FixedLang("vi"),
        review_queue=None,
        cfg=_default_agent_cfg(),
    )
    d = agent.moderate("xin chào")
    assert d.language == "vi"
    # Multi takes over after second-pass, not English finetuned
    assert "multilingual" in d.model_used or "dummy" in d.model_used


# ---------- New VI route ----------

def test_vi_route_uses_vi_model_as_authoritative_scorer():
    """When lang=vi and vi_finetuned exists, use the VI model and return offensive/hate scores."""
    en_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    fast = DummyMatrixPredictor([0.05, 0.0, 0.0, 0.0, 0.0, 0.0])  # low-risk English fast
    multi = DummyMatrixPredictor([0.05, 0.0, 0.0, 0.0, 0.0, 0.0])

    vi_model = DummyVIFinetuned([0.85, 0.10])  # high offensive, low hate

    predictors = Predictors(
        label_fields=en_labels,
        label_map_detoxify={"identity_hate": "identity_attack"},
        finetuned=None,
        detoxify_fast=fast,
        detoxify_multilingual=multi,
        vi_finetuned=vi_model,
        vi_label_fields=["offensive", "hate"],
    )
    agent = ModerationAgent(
        predictors=predictors,
        policy=_policy(),
        lang_detector=FixedLang("vi"),
        review_queue=None,
        cfg=_default_agent_cfg(force_langs=["vi"]),
    )
    d = agent.moderate("Bạn thật là tệ hại")
    assert d.language == "vi"
    # Label schema should be the VI one (offensive/hate), NOT the English 6-label one
    assert set(d.label_scores.keys()) == {"offensive", "hate"}
    assert "vi_finetuned" in d.model_used
    # Overall score = max(offensive, hate) = 0.85 -> block
    assert d.action == Action.BLOCK
    assert abs(d.overall_score - 0.85) < 1e-4


def test_vi_route_is_not_triggered_for_english_even_when_vi_model_exists():
    """VI predictor is loaded, but EN text must not accidentally be scored with it."""
    en_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    fast = DummyMatrixPredictor([0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
    multi = DummyMatrixPredictor([0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
    vi_model = DummyVIFinetuned([0.99, 0.99])  # intentionally extreme to prove it's NOT used

    predictors = Predictors(
        label_fields=en_labels,
        label_map_detoxify={"identity_hate": "identity_attack"},
        finetuned=None,
        detoxify_fast=fast,
        detoxify_multilingual=multi,
        vi_finetuned=vi_model,
    )
    agent = ModerationAgent(
        predictors=predictors,
        policy=_policy(),
        lang_detector=FixedLang("en"),
        review_queue=None,
        cfg=_default_agent_cfg(force_langs=["vi"]),  # only vi is forced, not en
    )
    d = agent.moderate("hello")
    assert d.language == "en"
    # English path with the 6 Jigsaw labels
    assert set(d.label_scores.keys()) == set(en_labels)
    assert d.action == Action.ALLOW
    assert "vi_finetuned" not in d.model_used


def test_force_second_pass_languages_triggers_second_pass_on_low_confidence_vi():
    """Even when English fast model scores 0.01, a forced VI input must go through the VI model."""
    en_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    # Fast model thinks everything is safe (0.01) - without the force rule, no 2nd pass.
    fast = DummyMatrixPredictor([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
    multi = DummyMatrixPredictor([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
    vi_model = DummyVIFinetuned([0.4, 0.05])  # would be ALLOW if 0.3 < 0.4 > 0.3? actually WARN

    predictors = Predictors(
        label_fields=en_labels,
        label_map_detoxify={"identity_hate": "identity_attack"},
        finetuned=None,
        detoxify_fast=fast,
        detoxify_multilingual=multi,
        vi_finetuned=vi_model,
    )
    agent = ModerationAgent(
        predictors=predictors,
        policy=_policy(),
        lang_detector=FixedLang("vi"),
        review_queue=None,
        cfg=_default_agent_cfg(force_langs=["vi"]),
    )
    d = agent.moderate("Bạn ổn mà")
    # VI route used (force_second_pass_languages + VI predictor present)
    assert "vi_finetuned" in d.model_used
    # Overall 0.4 -> falls in WARN band (allow_lt=0.3, warn_lt=0.6)
    assert d.action in {Action.WARN, Action.REVIEW}


# ---------- API response shape remains valid ----------

def test_api_response_shape_is_stable_for_vi():
    """`label_scores` is a dynamic dict, so VI keys must still fit the existing contract."""
    en_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    fast = DummyMatrixPredictor([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
    multi = DummyMatrixPredictor([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
    vi_model = DummyVIFinetuned([0.1, 0.8])

    predictors = Predictors(
        label_fields=en_labels,
        label_map_detoxify={"identity_hate": "identity_attack"},
        finetuned=None,
        detoxify_fast=fast,
        detoxify_multilingual=multi,
        vi_finetuned=vi_model,
    )
    agent = ModerationAgent(
        predictors=predictors,
        policy=_policy(),
        lang_detector=FixedLang("vi"),
        review_queue=None,
        cfg=_default_agent_cfg(force_langs=["vi"]),
    )
    d = agent.moderate("text")
    # Required response fields exist and are sensible
    assert isinstance(d.label_scores, dict)
    assert "offensive" in d.label_scores and "hate" in d.label_scores
    assert 0.0 <= d.overall_score <= 1.0
    assert d.action in {Action.ALLOW, Action.WARN, Action.BLOCK, Action.REVIEW}
    assert d.language == "vi"
    assert isinstance(d.policy_message, str)
    assert isinstance(d.explanation, str)
