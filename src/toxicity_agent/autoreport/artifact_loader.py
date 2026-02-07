from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_by_name(paths: list[Path]) -> Optional[Path]:
    if not paths:
        return None
    return sorted(paths, key=lambda p: p.name)[-1]


def find_latest_file(dir_path: Path, pattern: str) -> Optional[Path]:
    if not dir_path.exists():
        return None
    candidates = list(dir_path.glob(pattern))
    return _latest_by_name(candidates)


def find_latest_dir(dir_path: Path, prefix: str) -> Optional[Path]:
    if not dir_path.exists():
        return None
    candidates = [p for p in dir_path.glob(f"{prefix}*") if p.is_dir()]
    return _latest_by_name(candidates)


@dataclass
class ArtifactBundle:
    # core
    train_config: Dict[str, Any] | None
    infer_config: Dict[str, Any] | None

    # run artifacts
    eval_results: Dict[str, Any] | None
    tune_results: Dict[str, Any] | None
    benchmark: Dict[str, Any] | None
    error_analysis: Dict[str, Any] | None
    fairness: Dict[str, Any] | None

    # model artifacts
    model_metadata: Dict[str, Any] | None
    thresholds: Dict[str, Any] | None

    # paths (for reference)
    paths: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paths": self.paths,
            "train_config": self.train_config,
            "infer_config": self.infer_config,
            "eval_results": self.eval_results,
            "tune_results": self.tune_results,
            "benchmark": self.benchmark,
            "error_analysis": self.error_analysis,
            "fairness": self.fairness,
            "model_metadata": self.model_metadata,
            "thresholds": self.thresholds,
        }


def load_configs(train_config_path: Path | None, infer_config_path: Path | None) -> tuple[Dict[str, Any] | None, Dict[str, Any] | None]:
    train_cfg = None
    infer_cfg = None

    if train_config_path and train_config_path.exists():
        import yaml

        train_cfg = yaml.safe_load(train_config_path.read_text(encoding="utf-8"))
    if infer_config_path and infer_config_path.exists():
        import yaml

        infer_cfg = yaml.safe_load(infer_config_path.read_text(encoding="utf-8"))
    return train_cfg, infer_cfg


def collect_artifacts(
    *,
    runs_dir: Path,
    models_dir: Path,
    train_config_path: Path | None = None,
    infer_config_path: Path | None = None,
) -> ArtifactBundle:
    """Collect latest artifacts produced by the pipeline.

    This is designed to be forgiving:
    - if an artifact doesn't exist, the field is None.
    """
    train_cfg, infer_cfg = load_configs(train_config_path, infer_config_path)

    # Latest eval
    eval_dir = find_latest_dir(runs_dir, "eval-")
    eval_results = None
    if eval_dir:
        p = eval_dir / "eval_results.json"
        if p.exists():
            eval_results = _load_json(p)

    # Latest tuning
    tune_dir = find_latest_dir(runs_dir, "tune-")
    tune_results = None
    if tune_dir:
        p = tune_dir / "tune_results.json"
        if p.exists():
            tune_results = _load_json(p)

    # Latest benchmark
    bench_dir = runs_dir / "benchmarks"
    bench_file = find_latest_file(bench_dir, "benchmark-*.json")
    benchmark = _load_json(bench_file) if bench_file and bench_file.exists() else None

    # Latest error analysis
    err_dir = runs_dir / "error_analysis"
    err_file = find_latest_file(err_dir, "error-analysis-*.json")
    error_analysis = _load_json(err_file) if err_file and err_file.exists() else None

    # Latest fairness
    fair_dir = runs_dir / "fairness"
    fair_file = find_latest_file(fair_dir, "fairness-*.json")
    fairness = _load_json(fair_file) if fair_file and fair_file.exists() else None

    # Model metadata + thresholds (from finetuned/latest)
    finetuned_latest = models_dir / "finetuned" / "latest"
    meta_path = finetuned_latest / "model_metadata.json"
    model_metadata = _load_json(meta_path) if meta_path.exists() else None

    th_path = finetuned_latest / "thresholds_val.json"
    thresholds = _load_json(th_path) if th_path.exists() else None

    return ArtifactBundle(
        train_config=train_cfg,
        infer_config=infer_cfg,
        eval_results=eval_results,
        tune_results=tune_results,
        benchmark=benchmark,
        error_analysis=error_analysis,
        fairness=fairness,
        model_metadata=model_metadata,
        thresholds=thresholds,
        paths={
            "runs_dir": str(runs_dir),
            "models_dir": str(models_dir),
            "train_config_path": str(train_config_path) if train_config_path else "",
            "infer_config_path": str(infer_config_path) if infer_config_path else "",
        },
    )
