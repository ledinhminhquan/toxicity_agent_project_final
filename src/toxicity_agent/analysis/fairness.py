from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from ..config import load_config
from ..data.dataset import load_and_prepare_dataset
from ..logging_utils import get_logger
from ..models.baseline_detoxify import DetoxifyPredictor
from ..models.hf_model import HFPredictor
from ..training.metrics import evaluate_multilabel
from ..utils import resolve_paths

logger = get_logger(__name__)


@dataclass
class SliceResult:
    slice_name: str
    n: int
    prevalence: Dict[str, float]
    metrics: Dict[str, Any]
    tpr_per_label: Dict[str, float]
    fpr_per_label: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slice_name": self.slice_name,
            "n": self.n,
            "prevalence": self.prevalence,
            "metrics": self.metrics,
            "tpr_per_label": self.tpr_per_label,
            "fpr_per_label": self.fpr_per_label,
        }


def load_fairness_slices_config(path: Path) -> Dict[str, List[str]]:
    """Load identity-term groups from YAML/JSON.

    Expected schema:
        groups:
          gender:
            - woman
            - man
          religion:
            - muslim
            - christian
    """
    if not path.exists():
        raise FileNotFoundError(f"Fairness slices config not found: {path}")

    if path.suffix.lower() in {".yml", ".yaml"}:
        import yaml

        obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        obj = json.loads(path.read_text(encoding="utf-8"))

    groups = obj.get("groups")
    if not isinstance(groups, dict):
        raise ValueError("Fairness config must contain a top-level 'groups' mapping.")
    clean: Dict[str, List[str]] = {}
    for k, v in groups.items():
        if not isinstance(v, list):
            raise ValueError(f"Group '{k}' must be a list of terms.")
        terms = [str(t).strip() for t in v if str(t).strip()]
        clean[str(k)] = terms
    return clean


def _compile_group_patterns(groups: Dict[str, List[str]]) -> Dict[str, re.Pattern]:
    patterns: Dict[str, re.Pattern] = {}
    for group, terms in groups.items():
        if not terms:
            continue
        # whole-word match, case-insensitive
        escaped = [re.escape(t) for t in terms]
        pat = r"\b(" + "|".join(escaped) + r")\b"
        patterns[group] = re.compile(pat, flags=re.IGNORECASE)
    return patterns


def _slice_masks(texts: Sequence[str], groups: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
    patterns = _compile_group_patterns(groups)
    masks: Dict[str, np.ndarray] = {}
    any_mask = np.zeros((len(texts),), dtype=bool)

    for group, pat in patterns.items():
        mask = np.array([bool(pat.search(t)) for t in texts], dtype=bool)
        masks[group] = mask
        any_mask |= mask

    masks["any_identity_mention"] = any_mask
    masks["no_identity_mention"] = ~any_mask
    return masks


def _tpr_fpr(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> Tuple[float, float]:
    # y_true_bin and y_pred_bin are 0/1 arrays for a single label
    tp = int(((y_true_bin == 1) & (y_pred_bin == 1)).sum())
    fn = int(((y_true_bin == 1) & (y_pred_bin == 0)).sum())
    fp = int(((y_true_bin == 0) & (y_pred_bin == 1)).sum())
    tn = int(((y_true_bin == 0) & (y_pred_bin == 0)).sum())

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return float(tpr), float(fpr)


def evaluate_fairness_slices(
    *,
    train_config_path: str,
    fairness_slices_path: Path,
    model_kind: str = "finetuned",
    split: str = "test",
    threshold: float = 0.5,
    max_samples: int | None = None,
) -> Dict[str, Any]:
    """Evaluate performance on identity-mention slices.

    model_kind:
      - "finetuned" (default): use models/finetuned/latest
      - "detoxify-unbiased": use detoxify model_type="unbiased"
      - "detoxify-multilingual": use detoxify model_type="multilingual"
    """
    cfg = load_config(train_config_path)
    paths_cfg = cfg.get("paths", {})
    paths = resolve_paths(
        data_dir_cfg=str(paths_cfg.get("data_dir", "")),
        artifacts_dir_cfg=str(paths_cfg.get("artifacts_dir", "")),
    )

    groups = load_fairness_slices_config(fairness_slices_path)

    loaded = load_and_prepare_dataset(cfg)
    label_fields = loaded.label_fields

    ds = loaded.dataset.get(split)
    if ds is None:
        raise ValueError(f"Split '{split}' not found in dataset. Available: {list(loaded.dataset.keys())}")

    if max_samples is not None:
        ds = ds.select(range(min(int(max_samples), len(ds))))

    texts = ds["text"]
    y_true = np.array(ds["labels"], dtype=np.float32)

    # Predict probabilities
    probs: np.ndarray
    device = "cuda" if _cuda_available() else "cpu"

    if model_kind == "finetuned":
        finetuned_dir = paths.models_dir / "finetuned" / "latest"
        if not finetuned_dir.exists():
            raise FileNotFoundError(f"Fine-tuned model not found at {finetuned_dir}. Run training first.")
        predictor = HFPredictor(model_dir=finetuned_dir, device=device, max_length=int(cfg["model"]["max_length"]))
        probs = predictor.predict_proba_matrix(texts, label_order=label_fields, batch_size=64)
        model_info = {"kind": model_kind, "model_dir": str(finetuned_dir)}
    elif model_kind in {"detoxify-unbiased", "detoxify-multilingual"}:
        detox_type = "unbiased" if model_kind == "detoxify-unbiased" else "multilingual"
        detox = DetoxifyPredictor(model_type=detox_type, device=device)
        label_map = {"identity_hate": "identity_attack"}  # detoxify class name normalization
        probs = detox.predict_proba_matrix(texts, label_order=label_fields, label_map=label_map, batch_size=64)
        model_info = {"kind": model_kind, "detoxify_model_type": detox_type}
    else:
        raise ValueError(f"Unknown model_kind: {model_kind}")

    # Threshold -> predictions
    y_pred = (probs >= float(threshold)).astype(np.int32)

    masks = _slice_masks(texts, groups)
    slice_results: List[SliceResult] = []

    for slice_name, mask in masks.items():
        idx = np.where(mask)[0]
        if idx.size == 0:
            slice_results.append(
                SliceResult(
                    slice_name=slice_name,
                    n=0,
                    prevalence={lf: 0.0 for lf in label_fields},
                    metrics={"note": "empty slice"},
                    tpr_per_label={lf: 0.0 for lf in label_fields},
                    fpr_per_label={lf: 0.0 for lf in label_fields},
                )
            )
            continue

        yt = y_true[idx]
        yp = probs[idx]
        yb = y_pred[idx]

        prevalence = {lf: float(np.mean(yt[:, j])) for j, lf in enumerate(label_fields)}
        metrics = evaluate_multilabel(yt, yp, label_fields=label_fields, threshold=float(threshold)).to_dict()

        tpr = {}
        fpr = {}
        for j, lf in enumerate(label_fields):
            t, f = _tpr_fpr(yt[:, j].astype(np.int32), yb[:, j].astype(np.int32))
            tpr[lf] = t
            fpr[lf] = f

        slice_results.append(
            SliceResult(
                slice_name=slice_name,
                n=int(idx.size),
                prevalence=prevalence,
                metrics=metrics,
                tpr_per_label=tpr,
                fpr_per_label=fpr,
            )
        )

    # Compute fairness "gaps": compare each group slice to no_identity_mention
    ref = next((s for s in slice_results if s.slice_name == "no_identity_mention"), None)
    gaps: Dict[str, Any] = {}
    if ref and ref.n > 0:
        ref_tpr = ref.tpr_per_label
        ref_fpr = ref.fpr_per_label
        for s in slice_results:
            if s.slice_name in {"no_identity_mention"}:
                continue
            if s.n == 0:
                continue
            gaps[s.slice_name] = {
                "tpr_gap": {lf: float(s.tpr_per_label[lf] - ref_tpr[lf]) for lf in label_fields},
                "fpr_gap": {lf: float(s.fpr_per_label[lf] - ref_fpr[lf]) for lf in label_fields},
                "f1_micro_gap": float(s.metrics.get("f1_micro", 0.0) - ref.metrics.get("f1_micro", 0.0)),
                "auc_macro_gap": float(s.metrics.get("auc_macro", 0.0) - ref.metrics.get("auc_macro", 0.0)),
            }

    out: Dict[str, Any] = {
        "model": model_info,
        "dataset": {
            "name": cfg.get("dataset", {}).get("name"),
            "split": split,
            "n_samples": int(len(texts)),
        },
        "threshold": float(threshold),
        "groups": groups,
        "slices": [s.to_dict() for s in slice_results],
        "gaps_vs_no_identity": gaps,
        "notes": [
            "Slices are based on simple keyword matching. This is a heuristic, not a definitive demographic attribute.",
            "Do not use this as the only fairness assessment; consider human review, domain-specific identities, and other biases.",
        ],
    }
    return out


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def save_fairness_report(out_path: Path, report: Dict[str, Any]) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved fairness report to {out_path}")
    return out_path
