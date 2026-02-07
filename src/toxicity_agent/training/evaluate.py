from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np

from ..config import load_config
from ..logging_utils import get_logger
from ..utils import resolve_paths, set_global_seed
from ..data.dataset import load_and_prepare_dataset
from ..models.baseline_detoxify import DetoxifyPredictor
from ..models.hf_model import HFPredictor
from .baselines import train_tfidf_lr
from .metrics import evaluate_multilabel

logger = get_logger(__name__)


def _now_tag() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")


def _best_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def run_eval(config_path: str) -> Dict[str, Any]:
    """Evaluate baselines and (optionally) the fine-tuned model.

    Key stability improvements:
    - Batched inference for Detoxify and HF models to avoid OOM.
    - Optional max_eval_samples in config to cap evaluation size.
    """
    cfg = load_config(config_path)
    seed = int(cfg.get("project", {}).get("seed", 42))
    set_global_seed(seed)

    paths_cfg = cfg.get("paths", {})
    paths = resolve_paths(
        data_dir_cfg=str(paths_cfg.get("data_dir", "")),
        artifacts_dir_cfg=str(paths_cfg.get("artifacts_dir", "")),
    )

    loaded = load_and_prepare_dataset(cfg)
    label_fields = loaded.label_fields

    ds_train = loaded.dataset["train"]
    ds_test = loaded.dataset["test"] if len(loaded.dataset["test"]) > 0 else loaded.dataset["validation"]

    # Optional subsets
    max_train = cfg["dataset"].get("max_train_samples")
    max_eval = cfg["dataset"].get("max_eval_samples")
    if max_train is not None:
        ds_train = ds_train.select(range(min(int(max_train), len(ds_train))))
    if max_eval is not None:
        ds_test = ds_test.select(range(min(int(max_eval), len(ds_test))))

    train_texts = ds_train["text"]
    train_y = np.array(ds_train["labels"], dtype=np.float32)
    test_texts = ds_test["text"]
    test_y = np.array(ds_test["labels"], dtype=np.float32)

    results: Dict[str, Any] = {
        "label_fields": label_fields,
        "n_train": len(train_texts),
        "n_test": len(test_texts),
    }

    # -------------------------
    # Baseline 1: TF-IDF + Logistic Regression
    # -------------------------
    base_cfg = cfg.get("baselines", {})
    tfidf = train_tfidf_lr(
        train_texts=train_texts,
        train_labels=train_y,
        label_fields=label_fields,
        tfidf_max_features=int(base_cfg.get("tfidf_max_features", 80000)),
        tfidf_ngram_range=tuple(base_cfg.get("tfidf_ngram_range", [1, 2])),
        lr_C=float(base_cfg.get("lr_C", 4.0)),
    )
    tfidf_probs = tfidf.predict_proba_matrix(test_texts)
    tfidf_eval = evaluate_multilabel(test_y, tfidf_probs, label_fields, threshold=0.5).to_dict()
    baseline_path = paths.models_dir / "baselines" / "tfidf_lr.joblib"
    tfidf.save(baseline_path)
    results["baseline_tfidf_lr"] = {**tfidf_eval, "artifact_path": str(baseline_path)}

    # -------------------------
    # Baseline 2: Detoxify pre-trained
    # -------------------------
    detoxify_model_type = "unbiased"  # strong baseline
    detox = DetoxifyPredictor(model_type=detoxify_model_type, device=_best_device())
    detox_map = {"identity_hate": "identity_attack"}
    detox_bs = int(cfg.get("inference", {}).get("batch_size", 64))
    detox_probs = detox.predict_proba_matrix(
        test_texts,
        label_order=label_fields,
        label_map=detox_map,
        batch_size=detox_bs,
    )
    detox_eval = evaluate_multilabel(test_y, detox_probs, label_fields, threshold=0.5).to_dict()
    results["baseline_detoxify_unbiased"] = detox_eval

    # -------------------------
    # Fine-tuned model (if exists)
    # -------------------------
    finetuned_latest = paths.models_dir / "finetuned" / "latest"
    if finetuned_latest.exists():
        hf = HFPredictor(model_dir=finetuned_latest, device=_best_device(), max_length=int(cfg["model"]["max_length"]))
        hf_bs = int(cfg.get("inference", {}).get("batch_size", 64))
        hf_probs = hf.predict_proba_matrix(test_texts, label_order=label_fields, batch_size=hf_bs)
        hf_eval = evaluate_multilabel(test_y, hf_probs, label_fields, threshold=0.5).to_dict()
        results["finetuned_transformer"] = {**hf_eval, "model_dir": str(finetuned_latest)}
    else:
        results["finetuned_transformer"] = {"error": "models/finetuned/latest not found. Run training first."}

    out_dir = paths.runs_dir / f"eval-{_now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eval_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info(f"Evaluation saved to {out_path}")
    return results
