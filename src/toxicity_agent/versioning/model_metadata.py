from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from ..logging_utils import get_logger

logger = get_logger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_run(cmd: list[str]) -> Optional[str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
        return out if out else None
    except Exception:
        return None


def get_git_commit(repo_root: Path) -> Optional[str]:
    # Works only if repo is a git checkout.
    return _safe_run(["git", "-C", str(repo_root), "rev-parse", "HEAD"])


def get_git_is_dirty(repo_root: Path) -> Optional[bool]:
    status = _safe_run(["git", "-C", str(repo_root), "status", "--porcelain"])
    if status is None:
        return None
    return len(status.strip()) > 0


def get_env_summary() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            try:
                info["gpu_name"] = torch.cuda.get_device_name(0)
            except Exception:
                info["gpu_name"] = None
            try:
                info["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
            except Exception:
                info["bf16_supported"] = None
    except Exception:
        pass

    return info


@dataclass
class ModelMetadata:
    # Identity
    model_name: str
    model_type: str  # e.g., "finetuned_transformer"
    created_at_utc: str
    run_id: str

    # Code version
    git_commit: Optional[str]
    git_dirty: Optional[bool]

    # Data provenance (no raw data)
    dataset_name: str
    dataset_text_field: str
    dataset_label_fields: list[str]
    n_train: Optional[int] = None
    n_val: Optional[int] = None
    n_test: Optional[int] = None

    # Training settings (subset)
    training_config: Dict[str, Any] | None = None

    # Metrics snapshot (subset)
    best_metric_name: Optional[str] = None
    best_metric_value: Optional[float] = None
    test_metrics: Dict[str, Any] | None = None

    # Environment snapshot
    environment: Dict[str, Any] | None = None


def write_model_metadata(
    *,
    model_dir: Path,
    repo_root: Path,
    cfg: Dict[str, Any],
    run_id: str,
    test_metrics: Dict[str, Any] | None,
    best_metric_name: Optional[str] = None,
    best_metric_value: Optional[float] = None,
    n_train: Optional[int] = None,
    n_val: Optional[int] = None,
    n_test: Optional[int] = None,
    model_type: str = "finetuned_transformer",
) -> Path:
    """Write model metadata JSON next to model artifacts.

    This is intentionally privacy-preserving:
    - It contains configuration and aggregate counts/metrics only.
    - It never stores raw text or examples.

    Returns:
        Path to the written JSON file.
    """
    model_dir.mkdir(parents=True, exist_ok=True)

    dataset_cfg = cfg.get("dataset", {})
    training_cfg = cfg.get("training", {})

    metadata = ModelMetadata(
        model_name=str(cfg.get("model", {}).get("hf_model_name", "unknown")),
        model_type=model_type,
        created_at_utc=_utc_now_iso(),
        run_id=run_id,
        git_commit=get_git_commit(repo_root),
        git_dirty=get_git_is_dirty(repo_root),
        dataset_name=str(dataset_cfg.get("name", "unknown")),
        dataset_text_field=str(dataset_cfg.get("text_field", "unknown")),
        dataset_label_fields=list(dataset_cfg.get("label_fields", [])),
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        training_config={
            # Keep a compact subset (avoid dumping huge objects)
            "batch_size": training_cfg.get("batch_size"),
            "eval_batch_size": training_cfg.get("eval_batch_size"),
            "learning_rate": training_cfg.get("learning_rate"),
            "weight_decay": training_cfg.get("weight_decay"),
            "num_train_epochs": training_cfg.get("num_train_epochs"),
            "warmup_ratio": training_cfg.get("warmup_ratio"),
            "gradient_accumulation_steps": training_cfg.get("gradient_accumulation_steps"),
            "bf16": training_cfg.get("bf16"),
            "fp16": training_cfg.get("fp16"),
            "gradient_checkpointing": training_cfg.get("gradient_checkpointing"),
            "early_stopping": training_cfg.get("early_stopping"),
            "early_stopping_patience": training_cfg.get("early_stopping_patience"),
            "use_pos_weight": training_cfg.get("use_pos_weight"),
        },
        best_metric_name=best_metric_name,
        best_metric_value=best_metric_value,
        test_metrics=test_metrics or None,
        environment=get_env_summary(),
    )

    out_path = model_dir / "model_metadata.json"
    out_path.write_text(json.dumps(asdict(metadata), indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Wrote model metadata to {out_path}")
    return out_path
