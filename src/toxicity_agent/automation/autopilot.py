from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from ..analysis.error_analysis import run_error_analysis, save_error_analysis
from ..analysis.fairness import evaluate_fairness_slices, save_fairness_report
from ..analysis.latency import benchmark_callable, save_latency_report
from ..analysis.thresholds import threshold_search
from ..agent.factory import build_agent
from ..autoreport.artifact_loader import collect_artifacts
from ..autoreport.report_pdf import generate_report_pdf
from ..autoreport.slides_pptx import generate_slides_pptx
from ..config import load_config
from ..logging_utils import get_logger
from ..training.evaluate import run_eval
from ..training.train import run_train
from ..training.tune import run_tune
from ..utils import resolve_paths, get_repo_root
from ..grading.checklist import run_grading_checklist, save_grading_checklist

logger = get_logger(__name__)


def _ts() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")


@dataclass
class AutopilotResult:
    run_timestamp: str
    paths: Dict[str, str]
    outputs: Dict[str, str]
    notes: list[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_timestamp": self.run_timestamp,
            "paths": self.paths,
            "outputs": self.outputs,
            "notes": self.notes,
        }


def run_autopilot(
    *,
    train_config_path: str,
    infer_config_path: str,
    fairness_config_path: str,
    out_dir: str | None = None,
    do_tune: bool = False,
    tune_config_path: str | None = None,
    do_train: bool = True,
    do_eval: bool = True,
    do_threshold_search: bool = True,
    threshold_split: str = "validation",
    do_benchmark: bool = True,
    benchmark_n: int = 300,
    benchmark_warmup: int = 10,
    do_error_analysis: bool = True,
    error_split: str = "test",
    error_threshold: float = 0.5,
    do_fairness: bool = True,
    fairness_split: str = "test",
    fairness_threshold: float = 0.5,
    generate_report: bool = True,
    generate_slides: bool = True,
    report_title: str = "Hate Speech & Toxicity Detection System",
    report_author: str = "(fill) Student / Team",
) -> AutopilotResult:
    """Run the full pipeline and generate report + slides.

    This is intended for the "one button" workflow in Colab:
    - tune (optional)
    - train
    - eval
    - thresholds
    - benchmark
    - error analysis
    - fairness
    - generate report PDF + slide deck PPTX

    It is designed to be robust:
    - if a non-critical step fails, it records the error and continues when possible.
    - critical failures (e.g., training) will raise unless do_train=False.
    """
    ts = _ts()

    # Resolve paths from train config
    train_cfg = load_config(train_config_path)
    paths_cfg = train_cfg.get("paths", {})
    paths = resolve_paths(
        data_dir_cfg=str(paths_cfg.get("data_dir", "")),
        artifacts_dir_cfg=str(paths_cfg.get("artifacts_dir", "")),
    )

    submission_dir = Path(out_dir) if out_dir else (paths.artifacts_dir / "submission" / f"submission-{ts}")
    submission_dir.mkdir(parents=True, exist_ok=True)

    outputs: Dict[str, str] = {}
    notes: list[str] = []

    # 1) Tuning
    if do_tune:
        try:
            cfg_path = tune_config_path or train_config_path
            res = run_tune(cfg_path)
            # run_tune already writes tune_results.json under runs/tune-*
            outputs["tune"] = "ok"
            notes.append("Tuning completed.")
        except Exception as e:
            outputs["tune"] = "failed"
            notes.append(f"Tuning failed: {e}")

    # 2) Train
    if do_train:
        try:
            model_dir, train_res = run_train(train_config_path)
            outputs["train"] = "ok"
            outputs["finetuned_model_dir"] = str(model_dir)
            notes.append("Training completed.")
        except Exception as e:
            outputs["train"] = "failed"
            notes.append(f"Training failed: {e}")
            raise

    # 3) Eval
    if do_eval:
        try:
            res = run_eval(train_config_path)
            outputs["eval"] = "ok"
            notes.append("Evaluation completed.")
        except Exception as e:
            outputs["eval"] = "failed"
            notes.append(f"Evaluation failed: {e}")

    # 4) Threshold search
    if do_threshold_search:
        try:
            th = threshold_search(
                train_config_path=train_config_path,
                split=threshold_split,
                grid=None,
                max_samples=None,
                batch_size=64,
                out_path=None,
            )
            outputs["thresholds"] = th.get("saved_to", "")
            notes.append("Threshold search completed.")
        except Exception as e:
            outputs["thresholds"] = "failed"
            notes.append(f"Threshold search failed: {e}")

    # 5) Benchmark agent
    if do_benchmark:
        try:
            agent = build_agent(infer_config_path)
            # Use safe synthetic texts (do NOT sample dataset text).
            texts = ["This is a safe message. Please be respectful."] * int(benchmark_n)

            stats = benchmark_callable(agent.moderate, texts=texts, warmup=int(benchmark_warmup))
            report = {
                "mode": "agent_end_to_end",
                "stats": stats.to_dict(),
                "n": int(benchmark_n),
                "warmup": int(benchmark_warmup),
            }
            out_path = paths.runs_dir / "benchmarks" / f"benchmark-{ts}.json"
            save_latency_report(out_path, report)
            outputs["benchmark"] = str(out_path)
            notes.append("Benchmark completed.")
        except Exception as e:
            outputs["benchmark"] = "failed"
            notes.append(f"Benchmark failed: {e}")

    # 6) Error analysis
    if do_error_analysis:
        try:
            report = run_error_analysis(
                train_config_path=train_config_path,
                split=error_split,
                threshold=float(error_threshold),
                max_samples=None,
                model_kind="finetuned",
            )
            out_path = paths.runs_dir / "error_analysis" / f"error-analysis-{ts}.json"
            save_error_analysis(out_path, report)
            outputs["error_analysis"] = str(out_path)
            notes.append("Error analysis completed.")
        except Exception as e:
            outputs["error_analysis"] = "failed"
            notes.append(f"Error analysis failed: {e}")

    # 7) Fairness
    if do_fairness:
        try:
            report = evaluate_fairness_slices(
                train_config_path=train_config_path,
                fairness_slices_path=Path(fairness_config_path),
                model_kind="finetuned",
                split=fairness_split,
                threshold=float(fairness_threshold),
                max_samples=None,
            )
            out_path = paths.runs_dir / "fairness" / f"fairness-{ts}.json"
            save_fairness_report(out_path, report)
            outputs["fairness"] = str(out_path)
            notes.append("Fairness evaluation completed.")
        except Exception as e:
            outputs["fairness"] = "failed"
            notes.append(f"Fairness evaluation failed: {e}")

    # 8) Collect artifacts and generate report/slides
    bundle = collect_artifacts(
        runs_dir=paths.runs_dir,
        models_dir=paths.models_dir,
        train_config_path=Path(train_config_path),
        infer_config_path=Path(infer_config_path),
    )

    if generate_report:
        try:
            report_path = submission_dir / "report.pdf"
            generate_report_pdf(bundle=bundle, out_path=report_path, title=report_title, author=report_author)
            outputs["report_pdf"] = str(report_path)
            notes.append("Report PDF generated.")
        except Exception as e:
            outputs["report_pdf"] = "failed"
            notes.append(f"Report generation failed: {e}")

    if generate_slides:
        try:
            slides_path = submission_dir / "slides.pptx"
            generate_slides_pptx(bundle=bundle, out_path=slides_path, title=report_title, author_line=report_author)
            outputs["slides_pptx"] = str(slides_path)
            notes.append("Slides PPTX generated.")
        except Exception as e:
            outputs["slides_pptx"] = "failed"
            notes.append(f"Slides generation failed: {e}")

    # 9) Save a compact artifact snapshot (for reproducibility; no raw text)
    snapshot_path = submission_dir / "artifacts_snapshot.json"
    snapshot_path.write_text(json.dumps(bundle.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    outputs["artifact_snapshot"] = str(snapshot_path)

    # 10) Write a manifest (will be updated with grading checklist below)
    manifest_path = submission_dir / "submission_manifest.json"
    base_manifest: Dict[str, Any] = {
        "timestamp": ts,
        "outputs": outputs,
        "paths": {
            "artifacts_dir": str(paths.artifacts_dir),
            "models_dir": str(paths.models_dir),
            "runs_dir": str(paths.runs_dir),
            "submission_dir": str(submission_dir),
        },
        "notes": notes,
    }
    manifest_path.write_text(json.dumps(base_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    outputs["manifest"] = str(manifest_path)

    # 11) Create a first bundle zip (so checklist can verify it exists)
    bundle_zip = submission_dir / "submission_bundle.zip"
    import zipfile

    def _write_zip() -> None:
        with zipfile.ZipFile(bundle_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for fname in [
                "report.pdf",
                "slides.pptx",
                "submission_manifest.json",
                "artifacts_snapshot.json",
                "grading_checklist.json",
            ]:
                fpath = submission_dir / fname
                if fpath.exists():
                    z.write(fpath, arcname=fname)

    _write_zip()
    outputs["submission_zip"] = str(bundle_zip)

    # 12) Run grading checklist (rubric completeness checker) and embed into manifest
    try:
        checklist = run_grading_checklist(
            repo_root=get_repo_root(),
            submission_dir=submission_dir,
            bundle=bundle,
        )
    except Exception as e:
        checklist = {
            "overall_status": "FAIL",
            "score_estimate_100": 0.0,
            "max_score": 100.0,
            "items": [],
            "notes": [f"Checklist failed to run: {e}"],
        }
        notes.append(f"Checklist failed: {e}")

    checklist_path = submission_dir / "grading_checklist.json"
    save_grading_checklist(checklist_path, checklist)
    outputs["grading_checklist"] = str(checklist_path)

    # Update manifest with checklist + summary
    base_manifest["grading_checklist"] = checklist
    base_manifest["grading_summary"] = {
        "overall_status": checklist.get("overall_status"),
        "score_estimate_100": checklist.get("score_estimate_100"),
    }
    manifest_path.write_text(json.dumps(base_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    # Rebuild the bundle zip so it contains the updated manifest + checklist file
    _write_zip()


    return AutopilotResult(
        run_timestamp=ts,
        paths={
            "artifacts_dir": str(paths.artifacts_dir),
            "models_dir": str(paths.models_dir),
            "runs_dir": str(paths.runs_dir),
            "submission_dir": str(submission_dir),
        },
        outputs=outputs,
        notes=notes,
    )
