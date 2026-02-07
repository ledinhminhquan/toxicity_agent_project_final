from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ..config import load_config
from ..data.dataset import load_and_prepare_dataset
from ..logging_utils import get_logger
from ..models.hf_model import HFPredictor
from ..training.metrics import evaluate_multilabel
from ..utils import resolve_paths

logger = get_logger(__name__)


def threshold_search(
    *,
    train_config_path: str,
    split: str = "validation",
    grid: List[float] | None = None,
    max_samples: int | None = None,
    batch_size: int = 64,
    out_path: str | None = None,
) -> Dict[str, Any]:
    """Grid-search thresholds on a given split (default: validation).

    Outputs:
      - best_global (threshold, f1_micro, f1_macro)
      - best_per_label (threshold per label optimizing per-label F1)

    Notes:
    - We do NOT store raw text.
    - This assumes fine-tuned model is available at models/finetuned/latest.
    """
    if grid is None:
        grid = [round(x, 2) for x in np.linspace(0.05, 0.95, 19).tolist()]

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

    y_true = np.array(ds["labels"], dtype=np.float32)
    texts = ds["text"]

    finetuned_dir = paths.models_dir / "finetuned" / "latest"
    if not finetuned_dir.exists():
        raise FileNotFoundError(f"Fine-tuned model not found at {finetuned_dir}. Run training first.")

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = HFPredictor(model_dir=finetuned_dir, device=device, max_length=int(cfg["model"]["max_length"]))
    probs = predictor.predict_proba_matrix(texts, label_order=label_fields, batch_size=int(batch_size))

    # Per-label search (simple F1)
    best_per_label: Dict[str, Dict[str, float]] = {}
    for j, lf in enumerate(label_fields):
        best_f1 = -1.0
        best_t = 0.5
        yt = y_true[:, j]
        yp = probs[:, j]
        for t in grid:
            pred = (yp >= t).astype(np.int32)
            tp = int(((pred == 1) & (yt == 1)).sum())
            fp = int(((pred == 1) & (yt == 0)).sum())
            fn = int(((pred == 0) & (yt == 1)).sum())
            denom = (2 * tp + fp + fn)
            f1 = (2 * tp / denom) if denom > 0 else 0.0
            if f1 > best_f1:
                best_f1 = float(f1)
                best_t = float(t)
        best_per_label[lf] = {"threshold": best_t, "f1": best_f1}

    # Global threshold search by micro F1
    best_global = {"threshold": 0.5, "f1_micro": -1.0, "f1_macro": -1.0}
    for t in grid:
        res = evaluate_multilabel(y_true, probs, label_fields=label_fields, threshold=float(t)).to_dict()
        if float(res.get("f1_micro", -1.0)) > float(best_global["f1_micro"]):
            best_global = {
                "threshold": float(t),
                "f1_micro": float(res.get("f1_micro", 0.0)),
                "f1_macro": float(res.get("f1_macro", 0.0)),
            }

    out = {
        "split": split,
        "n_samples": int(len(texts)),
        "grid": list(grid),
        "best_global": best_global,
        "best_per_label": best_per_label,
        "label_fields": label_fields,
    }

    save_path = Path(out_path) if out_path else (finetuned_dir / "thresholds_val.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved thresholds to {save_path}")

    return {"saved_to": str(save_path), **out}
