from __future__ import annotations

import hashlib
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def set_global_seed(seed: int) -> None:
    """Best-effort seeding for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # noqa: F401
        import torch as _torch

        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
    except Exception:
        # Torch not installed / not available
        pass


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?(?:\(\d{2,4}\)[\s-]?)?\d{2,4}[\s-]?\d{2,4}[\s-]?\d{2,4}\b")


def redact_pii(text: str) -> str:
    """Basic PII redaction (best-effort).

    This is intentionally conservative and may over-redact some numeric sequences.
    """
    text = _EMAIL_RE.sub("<EMAIL>", text)
    text = _URL_RE.sub("<URL>", text)
    text = _PHONE_RE.sub("<PHONE>", text)
    return text


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_repo_root() -> Path:
    # Assumes this file is src/toxicity_agent/utils.py
    return Path(__file__).resolve().parents[2]


def env_path(key: str, default: Optional[str] = None) -> Optional[Path]:
    val = os.getenv(key, default)
    if val is None or val == "":
        return None
    return Path(val)


@dataclass(frozen=True)
class ResolvedPaths:
    repo_root: Path
    data_dir: Path
    artifacts_dir: Path
    models_dir: Path
    runs_dir: Path


def resolve_paths(data_dir_cfg: str = "", artifacts_dir_cfg: str = "") -> ResolvedPaths:
    """Resolve directories with a priority:

    1) explicit config value (if non-empty)
    2) environment variables
    3) repo defaults
    """
    repo_root = get_repo_root()

    # data_dir
    data_dir = Path(data_dir_cfg) if data_dir_cfg else (env_path("DATA_DIR") or (repo_root / "data"))
    artifacts_dir = (
        Path(artifacts_dir_cfg)
        if artifacts_dir_cfg
        else (env_path("ARTIFACTS_DIR") or (repo_root / "artifacts"))
    )

    models_dir = env_path("TOXICITY_MODEL_DIR") or (artifacts_dir / "models")
    runs_dir = env_path("TOXICITY_RUN_DIR") or (artifacts_dir / "runs")

    ensure_dir(data_dir)
    ensure_dir(artifacts_dir)
    ensure_dir(models_dir)
    ensure_dir(runs_dir)

    return ResolvedPaths(
        repo_root=repo_root,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        models_dir=models_dir,
        runs_dir=runs_dir,
    )
