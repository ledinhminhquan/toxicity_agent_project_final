from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from ..config import load_config
from ..logging_utils import get_logger
from ..models.model_registry import load_predictors
from ..utils import get_repo_root, resolve_paths
from ..agent.language import LanguageDetector
from ..agent.moderation_agent import ModerationAgent
from ..agent.policy_store import PolicyStore
from ..agent.human_review_queue import HumanReviewQueue
from ..monitoring.log_writer import PredictionLogger

logger = get_logger(__name__)

DEFAULT_LABEL_FIELDS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


def _infer_config_path() -> str:
    return os.getenv("TOXICITY_INFER_CONFIG", str(get_repo_root() / "configs" / "infer.yaml"))


@lru_cache(maxsize=1)
def get_infer_config() -> Dict[str, Any]:
    path = _infer_config_path()
    logger.info(f"Loading infer config: {path}")
    return load_config(path)


def _load_label_fields(models_dir: Path, finetuned_subdir: str) -> List[str]:
    lf_path = models_dir / finetuned_subdir / "label_fields.json"
    if lf_path.exists():
        try:
            return json.loads(lf_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return list(DEFAULT_LABEL_FIELDS)


@lru_cache(maxsize=1)
def get_agent() -> ModerationAgent:
    cfg = get_infer_config()
    paths = resolve_paths(
        data_dir_cfg=str(cfg.get("paths", {}).get("data_dir", "")),
        artifacts_dir_cfg=str(cfg.get("paths", {}).get("artifacts_dir", "")),
    )

    finetuned_subdir = cfg.get("inference", {}).get("finetuned_subdir", "finetuned/latest")
    label_fields = _load_label_fields(paths.models_dir, finetuned_subdir)

    predictors = load_predictors(cfg, label_fields=label_fields, models_dir=paths.models_dir)

    policy_path = get_repo_root() / "configs" / "policy_rules.yaml"
    policy = PolicyStore.load(policy_path)

    logging_cfg = cfg.get("logging", {})
    privacy_cfg = cfg.get("privacy", {})

    review_path = paths.runs_dir / str(logging_cfg.get("human_review_queue_name", "human_review_queue.jsonl"))
    review_queue = HumanReviewQueue(review_path)

    agent = ModerationAgent(
        predictors=predictors,
        policy=policy,
        lang_detector=LanguageDetector(),
        review_queue=review_queue,
        cfg=cfg,
    )
    return agent


@lru_cache(maxsize=1)
def get_prediction_logger() -> PredictionLogger:
    cfg = get_infer_config()
    paths = resolve_paths(
        data_dir_cfg=str(cfg.get("paths", {}).get("data_dir", "")),
        artifacts_dir_cfg=str(cfg.get("paths", {}).get("artifacts_dir", "")),
    )
    logging_cfg = cfg.get("logging", {})
    privacy_cfg = cfg.get("privacy", {})

    pred_path = paths.runs_dir / str(logging_cfg.get("predictions_log_name", "predictions.jsonl"))
    return PredictionLogger(
        path=pred_path,
        store_raw_text=bool(privacy_cfg.get("store_raw_text_in_logs", False)),
        store_redacted_text=bool(privacy_cfg.get("store_redacted_text_in_logs", False)),
    )
