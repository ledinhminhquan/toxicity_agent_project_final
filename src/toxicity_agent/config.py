from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

from .utils import get_repo_root


def load_env(env_path: Optional[str] = None) -> None:
    """Load .env if present."""
    if env_path:
        load_dotenv(env_path)
        return
    # default: repo root .env
    default = get_repo_root() / ".env"
    if default.exists():
        load_dotenv(str(default))


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def env_interpolate(value: Any) -> Any:
    """Interpolate ${VAR} in strings."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: env_interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [env_interpolate(v) for v in value]
    return value


def load_config(path_str: str) -> Dict[str, Any]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    cfg = load_yaml(path)
    return env_interpolate(cfg)
