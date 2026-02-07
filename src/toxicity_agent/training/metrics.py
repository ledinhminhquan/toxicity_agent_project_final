from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def binarize_probs(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (probs >= threshold).astype(np.int32)


def safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
    """Compute ROC-AUC if both classes present; otherwise return None."""
    try:
        # roc_auc_score can fail if y_true has only one class
        if len(np.unique(y_true)) < 2:
            return None
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return None


@dataclass
class EvalResult:
    f1_micro: float
    f1_macro: float
    auc_macro: Optional[float]
    auc_per_label: Dict[str, Optional[float]]

    def to_dict(self) -> Dict:
        return {
            "f1_micro": self.f1_micro,
            "f1_macro": self.f1_macro,
            "auc_macro": self.auc_macro,
            "auc_per_label": self.auc_per_label,
        }


def evaluate_multilabel(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_fields: Sequence[str],
    threshold: float = 0.5,
) -> EvalResult:
    y_pred = binarize_probs(y_prob, threshold=threshold)

    f1_micro = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    aucs: Dict[str, Optional[float]] = {}
    auc_values: List[float] = []
    for i, lf in enumerate(label_fields):
        auc_i = safe_roc_auc(y_true[:, i], y_prob[:, i])
        aucs[lf] = auc_i
        if auc_i is not None:
            auc_values.append(auc_i)

    auc_macro = float(np.mean(auc_values)) if auc_values else None

    return EvalResult(
        f1_micro=f1_micro,
        f1_macro=f1_macro,
        auc_macro=auc_macro,
        auc_per_label=aucs,
    )
