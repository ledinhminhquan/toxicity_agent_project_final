from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..logging_utils import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from .baseline_detoxify import DetoxifyPredictor  # noqa: F401
    from .hf_model import HFPredictor  # noqa: F401


@dataclass
class Predictors:
    label_fields: List[str]
    label_map_detoxify: Dict[str, str]
    finetuned: Optional[Any]
    detoxify_fast: Any
    detoxify_multilingual: Any


def _read_training_max_length(model_dir: Path) -> Optional[int]:
    """Read max_length from model_metadata.json to sync train/infer settings."""
    meta_path = model_dir / "model_metadata.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        tc = meta.get("training_config") or {}
        ml = tc.get("max_length")
        return int(ml) if ml is not None else None
    except Exception:
        return None


def load_predictors(
    cfg_infer: Dict,
    label_fields: List[str],
    models_dir: Path,
) -> Predictors:
    """Create predictor objects for inference.

    Notes:
    - Imports are intentionally *lazy* so unit tests that don't require
      transformers/detoxify can run fast.
    - If a finetuned model exists and its metadata records a different max_length
      than infer config, the model's training max_length takes precedence (avoids
      silent truncation mismatch).
    """
    from .baseline_detoxify import DetoxifyPredictor
    from .hf_model import HFPredictor

    infer = cfg_infer["inference"]
    detox = cfg_infer["detoxify"]

    device = infer.get("device", "cpu")
    max_length = int(infer.get("max_length", 256))

    label_map_detoxify = {"identity_hate": "identity_attack"}

    detoxify_fast = DetoxifyPredictor(model_type=detox["fast_model_type"], device=device)
    detoxify_multi = DetoxifyPredictor(model_type=detox["multilingual_model_type"], device=device)

    finetuned_subdir = infer.get("finetuned_subdir", "finetuned/latest")
    finetuned_dir = models_dir / finetuned_subdir
    finetuned = None
    if finetuned_dir.exists():
        train_ml = _read_training_max_length(finetuned_dir)
        if train_ml is not None and train_ml != max_length:
            logger.warning(
                "infer config max_length=%d but model was trained with max_length=%d; "
                "using the model's training value to avoid truncation mismatch.",
                max_length, train_ml,
            )
            max_length = train_ml
        finetuned = HFPredictor(model_dir=finetuned_dir, device=device, max_length=max_length, label_fields=label_fields)

    return Predictors(
        label_fields=label_fields,
        label_map_detoxify=label_map_detoxify,
        finetuned=finetuned,
        detoxify_fast=detoxify_fast,
        detoxify_multilingual=detoxify_multi,
    )
