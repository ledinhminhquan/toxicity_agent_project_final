from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    # Only for type checking; avoid importing heavy deps at runtime for lightweight tests.
    from .baseline_detoxify import DetoxifyPredictor  # noqa: F401
    from .hf_model import HFPredictor  # noqa: F401


@dataclass
class Predictors:
    label_fields: List[str]
    label_map_detoxify: Dict[str, str]
    finetuned: Optional[Any]
    detoxify_fast: Any
    detoxify_multilingual: Any


def load_predictors(
    cfg_infer: Dict,
    label_fields: List[str],
    models_dir: Path,
) -> Predictors:
    """Create predictor objects for inference.

    Notes:
    - Imports are intentionally *lazy* so unit tests that don't require transformers/detoxify can run fast.
    - In production usage (train/eval/serve), dependencies are installed and imports succeed.
    """
    # Lazy imports to avoid heavy dependency import at module import time.
    from .baseline_detoxify import DetoxifyPredictor
    from .hf_model import HFPredictor

    infer = cfg_infer["inference"]
    detox = cfg_infer["detoxify"]

    device = infer.get("device", "cpu")
    max_length = int(infer.get("max_length", 256))

    # label mapping for detoxify: some versions use identity_attack instead of identity_hate.
    # We keep the dataset label 'identity_hate' but map it during detoxify evaluation/inference.
    label_map_detoxify = {"identity_hate": "identity_attack"}

    detoxify_fast = DetoxifyPredictor(model_type=detox["fast_model_type"], device=device)
    detoxify_multi = DetoxifyPredictor(model_type=detox["multilingual_model_type"], device=device)

    finetuned_subdir = infer.get("finetuned_subdir", "finetuned/latest")
    finetuned_dir = models_dir / finetuned_subdir
    finetuned = None
    if finetuned_dir.exists():
        finetuned = HFPredictor(model_dir=finetuned_dir, device=device, max_length=max_length, label_fields=label_fields)

    return Predictors(
        label_fields=label_fields,
        label_map_detoxify=label_map_detoxify,
        finetuned=finetuned,
        detoxify_fast=detoxify_fast,
        detoxify_multilingual=detoxify_multi,
    )
