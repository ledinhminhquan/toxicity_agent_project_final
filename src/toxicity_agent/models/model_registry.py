from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..logging_utils import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from .baseline_detoxify import DetoxifyPredictor  # noqa: F401
    from .hf_model import HFPredictor  # noqa: F401


# Default Vietnamese sidecar label fields (2-label taxonomy, multi-label head).
# Kept here as a module-level constant so tests and CLI can import it.
VI_DEFAULT_LABEL_FIELDS: List[str] = ["offensive", "hate"]


@dataclass
class Predictors:
    """Container for all inference-time predictors.

    Backward-compatible: the Vietnamese fields are optional. If the VI sidecar
    model or its label file is missing, ``vi_finetuned`` stays ``None`` and the
    agent falls back to the original behaviour (Detoxify multilingual).
    """

    label_fields: List[str]
    label_map_detoxify: Dict[str, str]
    finetuned: Optional[Any]
    detoxify_fast: Any
    detoxify_multilingual: Any
    # Vietnamese sidecar (optional, additive-only)
    vi_finetuned: Optional[Any] = None
    vi_label_fields: List[str] = field(default_factory=lambda: list(VI_DEFAULT_LABEL_FIELDS))


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


def _read_label_fields(model_dir: Path, fallback: List[str]) -> List[str]:
    """Read label_fields.json if present, else return the provided fallback."""
    lf_path = model_dir / "label_fields.json"
    if not lf_path.exists():
        return list(fallback)
    try:
        labels = json.loads(lf_path.read_text(encoding="utf-8"))
        if isinstance(labels, list) and all(isinstance(x, str) for x in labels):
            return labels
    except Exception:
        pass
    return list(fallback)


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
    - Optional Vietnamese sidecar: if ``inference.vi_finetuned_subdir`` is set
      in the config **and** the corresponding folder exists, a VI predictor is
      also loaded. If not, the English pipeline is unchanged.
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

    # --------- Vietnamese sidecar (optional, additive) ---------
    vi_finetuned = None
    vi_label_fields = list(VI_DEFAULT_LABEL_FIELDS)
    vi_subdir = infer.get("vi_finetuned_subdir")
    if vi_subdir:
        vi_dir = models_dir / vi_subdir
        if vi_dir.exists():
            vi_label_fields = _read_label_fields(vi_dir, VI_DEFAULT_LABEL_FIELDS)
            vi_max_length = int(infer.get("vi_max_length", 256))
            vi_train_ml = _read_training_max_length(vi_dir)
            if vi_train_ml is not None and vi_train_ml != vi_max_length:
                logger.warning(
                    "infer config vi_max_length=%d but VI model was trained with max_length=%d; "
                    "using the model's training value.",
                    vi_max_length, vi_train_ml,
                )
                vi_max_length = vi_train_ml
            try:
                vi_finetuned = HFPredictor(
                    model_dir=vi_dir,
                    device=device,
                    max_length=vi_max_length,
                    label_fields=vi_label_fields,
                )
                logger.info("Loaded Vietnamese sidecar predictor from %s", vi_dir)
            except Exception as e:
                # Never fail inference if VI sidecar can't load - just skip it.
                logger.warning("Could not load Vietnamese sidecar predictor: %s", e)
                vi_finetuned = None
        else:
            logger.info(
                "VI sidecar subdir configured (%s) but folder missing (%s); skipping.",
                vi_subdir, vi_dir,
            )

    return Predictors(
        label_fields=label_fields,
        label_map_detoxify=label_map_detoxify,
        finetuned=finetuned,
        detoxify_fast=detoxify_fast,
        detoxify_multilingual=detoxify_multi,
        vi_finetuned=vi_finetuned,
        vi_label_fields=vi_label_fields,
    )
