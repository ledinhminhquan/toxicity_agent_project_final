from __future__ import annotations

import json
from pathlib import Path

from ..config import load_config
from ..models.model_registry import load_predictors
from ..utils import get_repo_root, resolve_paths
from .human_review_queue import HumanReviewQueue
from .language import LanguageDetector
from .moderation_agent import ModerationAgent
from .policy_store import PolicyStore


def build_agent(config_path: str) -> ModerationAgent:
    """Build a ModerationAgent from an inference config.

    This is used by both CLI and the autopilot pipeline.
    """
    cfg = load_config(config_path)
    paths = resolve_paths(
        data_dir_cfg=str(cfg.get("paths", {}).get("data_dir", "")),
        artifacts_dir_cfg=str(cfg.get("paths", {}).get("artifacts_dir", "")),
    )

    finetuned_subdir = cfg.get("inference", {}).get("finetuned_subdir", "finetuned/latest")
    # Load labels if available
    label_fields = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    lf_path = paths.models_dir / finetuned_subdir / "label_fields.json"
    if lf_path.exists():
        try:
            label_fields = json.loads(lf_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    predictors = load_predictors(cfg, label_fields=label_fields, models_dir=paths.models_dir)
    policy = PolicyStore.load(get_repo_root() / "configs" / "policy_rules.yaml")

    review_path = paths.runs_dir / str(cfg.get("logging", {}).get("human_review_queue_name", "human_review_queue.jsonl"))
    review_queue = HumanReviewQueue(review_path)

    agent = ModerationAgent(
        predictors=predictors,
        policy=policy,
        lang_detector=LanguageDetector(),
        review_queue=review_queue,
        cfg=cfg,
    )
    return agent
