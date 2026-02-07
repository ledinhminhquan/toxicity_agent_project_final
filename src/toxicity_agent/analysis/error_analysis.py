from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from ..config import load_config
from ..data.dataset import load_and_prepare_dataset
from ..logging_utils import get_logger
from ..models.baseline_detoxify import DetoxifyPredictor
from ..models.hf_model import HFPredictor
from ..training.metrics import evaluate_multilabel
from ..utils import resolve_paths, sha256_text

logger = get_logger(__name__)


_RE_URL = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_RE_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", flags=re.IGNORECASE)
_RE_REPEAT = re.compile(r"(.)\1{2,}")
_PUNCT = set("!?.:,;\"'()[]{}<>/\\|@#$%^&*_+-=~`")


@dataclass
class ErrorAnalysisReport:
    label_fields: List[str]
    n_samples: int
    threshold: float
    model: Dict[str, Any]
    overall_metrics: Dict[str, Any]
    confusion_per_label: Dict[str, Any]
    feature_summary: Dict[str, Any]
    top_error_cases: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label_fields": self.label_fields,
            "n_samples": self.n_samples,
            "threshold": self.threshold,
            "model": self.model,
            "overall_metrics": self.overall_metrics,
            "confusion_per_label": self.confusion_per_label,
            "feature_summary": self.feature_summary,
            "top_error_cases": self.top_error_cases,
        }


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _extract_features(texts: Sequence[str]) -> Dict[str, np.ndarray]:
    n = len(texts)
    length_chars = np.zeros((n,), dtype=np.int32)
    length_words = np.zeros((n,), dtype=np.int32)
    uppercase_ratio = np.zeros((n,), dtype=np.float32)
    punct_ratio = np.zeros((n,), dtype=np.float32)
    exclam = np.zeros((n,), dtype=np.int32)
    question = np.zeros((n,), dtype=np.int32)
    has_url = np.zeros((n,), dtype=np.int8)
    has_email = np.zeros((n,), dtype=np.int8)
    has_repeat = np.zeros((n,), dtype=np.int8)
    non_ascii_ratio = np.zeros((n,), dtype=np.float32)
    has_quote = np.zeros((n,), dtype=np.int8)

    for i, t in enumerate(texts):
        s = t or ""
        length_chars[i] = len(s)
        words = s.split()
        length_words[i] = len(words)

        if len(s) > 0:
            upper = sum(1 for c in s if c.isalpha() and c.isupper())
            alpha = sum(1 for c in s if c.isalpha())
            uppercase_ratio[i] = (upper / alpha) if alpha > 0 else 0.0

            punct = sum(1 for c in s if c in _PUNCT)
            punct_ratio[i] = punct / len(s)

            non_ascii = sum(1 for c in s if ord(c) > 127)
            non_ascii_ratio[i] = non_ascii / len(s)

        exclam[i] = s.count("!")
        question[i] = s.count("?")
        has_url[i] = 1 if _RE_URL.search(s) else 0
        has_email[i] = 1 if _RE_EMAIL.search(s) else 0
        has_repeat[i] = 1 if _RE_REPEAT.search(s) else 0
        has_quote[i] = 1 if ('"' in s or "'" in s) else 0

    return {
        "length_chars": length_chars,
        "length_words": length_words,
        "uppercase_ratio": uppercase_ratio,
        "punct_ratio": punct_ratio,
        "exclam": exclam,
        "question": question,
        "has_url": has_url,
        "has_email": has_email,
        "has_repeat": has_repeat,
        "non_ascii_ratio": non_ascii_ratio,
        "has_quote": has_quote,
    }


def _summarize_features(features: Dict[str, np.ndarray], mask: np.ndarray) -> Dict[str, Any]:
    """Return a compact numeric summary of features for subset mask."""
    idx = np.where(mask)[0]
    if idx.size == 0:
        return {"n": 0}

    out: Dict[str, Any] = {"n": int(idx.size)}
    for k, arr in features.items():
        sub = arr[idx]
        # Use robust stats
        out[k] = {
            "mean": float(np.mean(sub)),
            "median": float(np.median(sub)),
            "p90": float(np.percentile(sub, 90)),
        }
    return out


def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    # y_true/y_pred are 0/1 arrays for one label
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def run_error_analysis(
    *,
    train_config_path: str,
    split: str = "test",
    threshold: float = 0.5,
    max_samples: int | None = None,
    model_kind: str = "finetuned",
    top_k_cases: int = 50,
) -> Dict[str, Any]:
    """Run privacy-preserving error analysis.

    - No raw text is printed or written.
    - We only store hashed ids + numeric features for a few top error cases.

    model_kind:
      - "finetuned" (default) -> models/finetuned/latest
      - "detoxify-unbiased" -> detoxify baseline
    """
    cfg = load_config(train_config_path)
    paths_cfg = cfg.get("paths", {})
    paths = resolve_paths(
        data_dir_cfg=str(paths_cfg.get("data_dir", "")),
        artifacts_dir_cfg=str(paths_cfg.get("artifacts_dir", "")),
    )

    loaded = load_and_prepare_dataset(cfg)
    label_fields = loaded.label_fields

    ds = loaded.dataset.get(split)
    if ds is None:
        raise ValueError(f"Split '{split}' not found. Available: {list(loaded.dataset.keys())}")
    if max_samples is not None:
        ds = ds.select(range(min(int(max_samples), len(ds))))

    texts = ds["text"]
    y_true = np.array(ds["labels"], dtype=np.float32)

    device = "cuda" if _cuda_available() else "cpu"
    probs: np.ndarray
    model_info: Dict[str, Any]

    if model_kind == "finetuned":
        model_dir = paths.models_dir / "finetuned" / "latest"
        if not model_dir.exists():
            raise FileNotFoundError(f"Fine-tuned model not found: {model_dir}. Run training first.")
        predictor = HFPredictor(model_dir=model_dir, device=device, max_length=int(cfg["model"]["max_length"]))
        probs = predictor.predict_proba_matrix(texts, label_order=label_fields, batch_size=64)
        model_info = {"kind": model_kind, "model_dir": str(model_dir)}
    elif model_kind == "detoxify-unbiased":
        detox = DetoxifyPredictor(model_type="unbiased", device=device)
        label_map = {"identity_hate": "identity_attack"}
        probs = detox.predict_proba_matrix(texts, label_order=label_fields, label_map=label_map, batch_size=64)
        model_info = {"kind": model_kind, "detoxify_model_type": "unbiased"}
    else:
        raise ValueError(f"Unknown model_kind: {model_kind}")

    y_pred = (probs >= float(threshold)).astype(np.int32)
    overall = evaluate_multilabel(y_true, probs, label_fields, threshold=float(threshold)).to_dict()

    # Per-label confusion
    confusion = {}
    for j, lf in enumerate(label_fields):
        confusion[lf] = _confusion_counts(y_true[:, j].astype(np.int32), y_pred[:, j].astype(np.int32))

    # Feature extraction
    feats = _extract_features(texts)

    # Overall error mask: any label differs
    any_error = np.any(y_pred != y_true.astype(np.int32), axis=1)

    # For each label, fp and fn masks
    per_label_summaries: Dict[str, Any] = {}
    for j, lf in enumerate(label_fields):
        yt = y_true[:, j].astype(np.int32)
        yp = y_pred[:, j].astype(np.int32)
        fp = (yt == 0) & (yp == 1)
        fn = (yt == 1) & (yp == 0)
        per_label_summaries[lf] = {
            "fp_features": _summarize_features(feats, fp),
            "fn_features": _summarize_features(feats, fn),
        }

    feature_summary = {
        "all_samples": _summarize_features(feats, np.ones((len(texts),), dtype=bool)),
        "any_error": _summarize_features(feats, any_error),
        "per_label": per_label_summaries,
        "notes": [
            "Feature summaries are numeric aggregates (mean/median/p90).",
            "No raw text is stored. 'top_error_cases' uses sha256(text) only.",
        ],
    }

    # Top error cases: choose by max absolute error (|prob - true|)
    # This provides a stable 'most wrong' set without exposing text.
    abs_err = np.abs(probs - y_true)
    # score per row: max over labels
    row_score = abs_err.max(axis=1)
    top_idx = np.argsort(-row_score)[: int(top_k_cases)]

    top_cases: List[Dict[str, Any]] = []
    for i in top_idx:
        t = texts[i]
        # hashed id only
        case = {
            "sha256": sha256_text(t),
            "length_chars": int(feats["length_chars"][i]),
            "length_words": int(feats["length_words"][i]),
            "has_url": int(feats["has_url"][i]),
            "has_email": int(feats["has_email"][i]),
            "has_quote": int(feats["has_quote"][i]),
            "uppercase_ratio": float(feats["uppercase_ratio"][i]),
            "max_abs_error": float(row_score[i]),
            "true_labels": {lf: int(y_true[i, j] > 0.5) for j, lf in enumerate(label_fields)},
            "pred_labels": {lf: int(y_pred[i, j]) for j, lf in enumerate(label_fields)},
            "pred_probs": {lf: float(probs[i, j]) for j, lf in enumerate(label_fields)},
        }
        top_cases.append(case)

    report = ErrorAnalysisReport(
        label_fields=label_fields,
        n_samples=int(len(texts)),
        threshold=float(threshold),
        model=model_info,
        overall_metrics=overall,
        confusion_per_label=confusion,
        feature_summary=feature_summary,
        top_error_cases=top_cases,
    )
    return report.to_dict()


def save_error_analysis(out_path: Path, report: Dict[str, Any]) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved error analysis report to {out_path}")
    return out_path
