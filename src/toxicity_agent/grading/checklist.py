from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..autoreport.artifact_loader import ArtifactBundle
from ..logging_utils import get_logger

logger = get_logger(__name__)


Status = str  # "PASS" | "WARN" | "FAIL"


@dataclass
class ChecklistItem:
    id: str
    title: str
    rubric_ref: str
    status: Status
    points: float
    max_points: float
    evidence: Dict[str, Any]
    fix_hint: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "rubric_ref": self.rubric_ref,
            "status": self.status,
            "points": self.points,
            "max_points": self.max_points,
            "evidence": self.evidence,
            "fix_hint": self.fix_hint,
        }


def _count_pdf_pages(pdf_path: Path) -> Optional[int]:
    """Count pages in a PDF without extra deps.

    Works well for reportlab-generated PDFs. Not guaranteed for all PDFs.
    """
    try:
        data = pdf_path.read_bytes()
        # Matches "/Type /Page" but not "/Type /Pages"
        matches = re.findall(rb"/Type\s*/Page\b", data)
        n = len(matches)
        return int(n) if n > 0 else None
    except Exception:
        return None


def _count_pptx_slides(pptx_path: Path) -> Optional[int]:
    try:
        from pptx import Presentation

        prs = Presentation(str(pptx_path))
        return int(len(prs.slides))
    except Exception:
        return None




def _zip_list(zip_path: Path) -> List[str]:
    try:
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as z:
            return z.namelist()
    except Exception:
        return []


def _zip_contains_required(zip_path: Path, required: List[str]) -> Tuple[bool, List[str]]:
    names = _zip_list(zip_path)
    missing = [r for r in required if r not in names]
    return (len(missing) == 0), missing

def _walk(obj: Any) -> Iterable[Any]:
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)
    else:
        yield obj


def _contains_raw_text_field(obj: Any) -> bool:
    """Best-effort check for raw text leakage in error analysis artifacts."""
    if not isinstance(obj, dict):
        return False
    suspicious_keys = {"text", "raw_text", "comment_text", "message", "content"}
    for k, v in obj.items():
        if k in suspicious_keys and isinstance(v, str) and v.strip():
            return True
        if isinstance(v, (dict, list)) and _contains_raw_text_field(v):
            return True
    return False


def _score_from_status(status: Status, max_points: float) -> float:
    if status == "PASS":
        return float(max_points)
    if status == "WARN":
        return float(max_points) * 0.5
    return 0.0


def _exists(path: Path) -> bool:
    try:
        return path.exists()
    except Exception:
        return False


def run_grading_checklist(
    *,
    repo_root: Path,
    submission_dir: Path,
    bundle: ArtifactBundle,
    expected_report_pages: Tuple[int, int] = (10, 15),
    expected_slides: Tuple[int, int] = (10, 15),
) -> Dict[str, Any]:
    """Run a rubric-oriented checklist and return structured results.

    Notes:
    - This is an automated *completeness* checker, not a substitute for instructor grading.
    - It tries to map to assignment rubric sections: docs, pipeline, deployment, agentic, ethics.
    """
    items: List[ChecklistItem] = []

    # -------------------------
    # Submission components
    # -------------------------
    report_pdf = submission_dir / "report.pdf"
    slides_pptx = submission_dir / "slides.pptx"
    bundle_zip = submission_dir / "submission_bundle.zip"

    pdf_pages = _count_pdf_pages(report_pdf) if _exists(report_pdf) else None
    pptx_slides = _count_pptx_slides(slides_pptx) if _exists(slides_pptx) else None

    status_pdf: Status
    if not _exists(report_pdf):
        status_pdf = "FAIL"
    elif pdf_pages is None:
        status_pdf = "WARN"
    else:
        status_pdf = "PASS" if (expected_report_pages[0] <= pdf_pages <= expected_report_pages[1]) else "FAIL"

    items.append(
        ChecklistItem(
            id="SUB_REPORT",
            title="Written report PDF exists and length is 10–15 pages",
            rubric_ref="II.1 (Report format/length)",
            status=status_pdf,
            max_points=10.0,
            points=_score_from_status(status_pdf, 10.0),
            evidence={"path": str(report_pdf), "page_count": pdf_pages, "expected": list(expected_report_pages)},
            fix_hint="Re-run autopilot report generation. Ensure report generator produces 10–15 pages." if status_pdf != "PASS" else None,
        )
    )

    status_slides: Status
    if not _exists(slides_pptx):
        status_slides = "FAIL"
    elif pptx_slides is None:
        status_slides = "WARN"
    else:
        status_slides = "PASS" if (expected_slides[0] <= pptx_slides <= expected_slides[1]) else "FAIL"

    items.append(
        ChecklistItem(
            id="SUB_SLIDES",
            title="Slide deck exists and length is 10–15 slides",
            rubric_ref="II.3 (Slides format/length)",
            status=status_slides,
            max_points=6.0,
            points=_score_from_status(status_slides, 6.0),
            evidence={"path": str(slides_pptx), "slide_count": pptx_slides, "expected": list(expected_slides)},
            fix_hint="Re-run autopilot slide generation. Ensure slide count 10–15." if status_slides != "PASS" else None,
        )
    )

    # Check bundle zip existence + required contents
    required_in_zip = ["report.pdf", "slides.pptx", "submission_manifest.json", "artifacts_snapshot.json"]
    if not _exists(bundle_zip):
        status_zip: Status = "FAIL"
        zip_ok = False
        zip_missing = required_in_zip
        zip_names = []
    else:
        zip_ok, zip_missing = _zip_contains_required(bundle_zip, required_in_zip)
        zip_names = _zip_list(bundle_zip)
        status_zip = "PASS" if zip_ok else "WARN"

    items.append(
        ChecklistItem(
            id="SUB_BUNDLE",
            title="Submission bundle zip exists (and contains required files)",
            rubric_ref="II (Submission components)",
            status=status_zip,
            max_points=2.0,
            points=_score_from_status(status_zip, 2.0),
            evidence={"path": str(bundle_zip), "required": required_in_zip, "missing": zip_missing, "zip_namelist_sample": zip_names[:20]},
            fix_hint="Re-run autopilot to rebuild submission_bundle.zip (must include report.pdf, slides.pptx, submission_manifest.json, artifacts_snapshot.json)." if status_zip == "FAIL" else None,
        )
    )

    # -------------------------
    # Repo structure & docs
    # -------------------------
    required_dirs = ["src", "configs", "tests", "data", "models"]
    required_files = ["README.md"]
    has_req = (repo_root / "requirements.txt").exists() or (repo_root / "pyproject.toml").exists()

    missing_dirs = [d for d in required_dirs if not (repo_root / d).exists()]
    missing_files = [f for f in required_files if not (repo_root / f).exists()]

    status_repo: Status = "PASS" if (not missing_dirs and not missing_files and has_req) else "FAIL"
    items.append(
        ChecklistItem(
            id="REPO_STRUCTURE",
            title="Repository structure + dependency file",
            rubric_ref="I.3 + II.2 (Repo structure/requirements/README)",
            status=status_repo,
            max_points=6.0,
            points=_score_from_status(status_repo, 6.0),
            evidence={
                "repo_root": str(repo_root),
                "missing_dirs": missing_dirs,
                "missing_files": missing_files,
                "has_requirements_or_pyproject": bool(has_req),
            },
            fix_hint="Ensure required folders/files exist: src/, data/, models/, configs/, tests/, README.md, requirements.txt or pyproject.toml." if status_repo != "PASS" else None,
        )
    )

    docs_required = {
        "problem_definition": repo_root / "docs" / "problem_definition.md",
        "data_description": repo_root / "docs" / "data_description.md",
        "project_plan": repo_root / "docs" / "project_plan.md",
        "continual_learning_monitoring": repo_root / "docs" / "continual_learning_monitoring.md",
        "privacy_robustness": repo_root / "docs" / "privacy_robustness.md",
        "ethics_statement": repo_root / "docs" / "ethics_statement.md",
        "model_versioning": repo_root / "docs" / "model_versioning.md",
    }
    missing_docs = [k for k, p in docs_required.items() if not p.exists()]

    status_docs: Status = "PASS" if not missing_docs else "FAIL"
    items.append(
        ChecklistItem(
            id="DOCS_CORE",
            title="Core documentation files exist (problem, data, plan, monitoring, privacy, ethics, versioning)",
            rubric_ref="I.2, I.4, I.8, I.9, I.10, I.11, I.6",
            status=status_docs,
            max_points=20.0,
            points=_score_from_status(status_docs, 20.0),
            evidence={"missing_docs": missing_docs, "docs": {k: str(p) for k, p in docs_required.items()}},
            fix_hint="Add missing docs under docs/ to match rubric deliverables." if status_docs != "PASS" else None,
        )
    )

    # -------------------------
    # Data + model + evaluation artifacts
    # -------------------------
    eval_res = bundle.eval_results
    tune_res = bundle.tune_results
    err = bundle.error_analysis
    fair = bundle.fairness
    bench = bundle.benchmark
    meta = bundle.model_metadata
    th = bundle.thresholds

    # Tuning required by rubric
    status_tune: Status = "PASS" if isinstance(tune_res, dict) and tune_res.get("best") else "FAIL"
    items.append(
        ChecklistItem(
            id="ML_TUNING",
            title="Hyperparameter tuning results exist (basic tuning required)",
            rubric_ref="I.5 (basic hyperparameter tuning)",
            status=status_tune,
            max_points=6.0,
            points=_score_from_status(status_tune, 6.0),
            evidence={"tune_results_found": bool(tune_res), "best": (tune_res or {}).get("best") if isinstance(tune_res, dict) else None},
            fix_hint="Run tuning (toxicity-agent tune) or autopilot with --do-tune." if status_tune != "PASS" else None,
        )
    )

    # Baseline comparison required
    status_eval: Status
    if not isinstance(eval_res, dict):
        status_eval = "FAIL"
    else:
        ok = all(k in eval_res for k in ["baseline_tfidf_lr", "baseline_detoxify_unbiased", "finetuned_transformer"])
        status_eval = "PASS" if ok else "FAIL"
    items.append(
        ChecklistItem(
            id="ML_EVAL_BASELINES",
            title="Evaluation exists with baseline comparison + fine-tuned model",
            rubric_ref="I.5 (baseline comparison + evaluation results)",
            status=status_eval,
            max_points=12.0,
            points=_score_from_status(status_eval, 12.0),
            evidence={"eval_keys": list(eval_res.keys()) if isinstance(eval_res, dict) else None},
            fix_hint="Run evaluation: toxicity-agent eval --config configs/train_final.yaml" if status_eval != "PASS" else None,
        )
    )

    # Error analysis required (and privacy-safe)
    status_err: Status
    leak = False
    if not isinstance(err, dict):
        status_err = "FAIL"
    else:
        leak = _contains_raw_text_field(err)
        status_err = "FAIL" if leak else "PASS"
    items.append(
        ChecklistItem(
            id="ML_ERROR_ANALYSIS",
            title="Error analysis exists and does NOT leak raw toxic text",
            rubric_ref="I.5 (error analysis) + I.9 (privacy)",
            status=status_err,
            max_points=8.0,
            points=_score_from_status(status_err, 8.0),
            evidence={"error_analysis_found": bool(err), "raw_text_leak_detected": bool(leak)},
            fix_hint="Ensure error analysis stores only hashed ids + aggregates, not raw text." if status_err != "PASS" else None,
        )
    )

    # Fairness evaluation (part of ethics)
    status_fair: Status = "PASS" if isinstance(fair, dict) and fair.get("slices") else "FAIL"
    items.append(
        ChecklistItem(
            id="ETHICS_FAIRNESS",
            title="Fairness slice evaluation exists (identity mentions)",
            rubric_ref="I.11 (bias/fairness risks) + I.5 (analysis)",
            status=status_fair,
            max_points=8.0,
            points=_score_from_status(status_fair, 8.0),
            evidence={"fairness_found": bool(fair), "has_slices": bool((fair or {}).get("slices") if isinstance(fair, dict) else False)},
            fix_hint="Run fairness eval: toxicity-agent fairness --config configs/train_final.yaml --fairness-config configs/fairness_slices.yaml" if status_fair != "PASS" else None,
        )
    )

    # Latency benchmark (deployment consideration)
    status_bench: Status = "PASS" if isinstance(bench, dict) and (bench.get("stats") or {}).get("p95_ms") is not None else "FAIL"
    items.append(
        ChecklistItem(
            id="DEPLOY_LATENCY",
            title="Latency benchmark exists (p50/p95/p99)",
            rubric_ref="I.6 (latency/scalability)",
            status=status_bench,
            max_points=4.0,
            points=_score_from_status(status_bench, 4.0),
            evidence={"benchmark_found": bool(bench), "stats": (bench or {}).get("stats") if isinstance(bench, dict) else None},
            fix_hint="Run benchmark: toxicity-agent benchmark --config configs/infer.yaml" if status_bench != "PASS" else None,
        )
    )

    # Threshold calibration is a strong practical addition (not strictly required, but good)
    status_th: Status = "PASS" if isinstance(th, dict) and (th.get("best_global") or th.get("best_global", {})) else "WARN"
    items.append(
        ChecklistItem(
            id="ML_THRESHOLDS",
            title="Threshold search / calibration available",
            rubric_ref="I.5 (model optimization) / practical deployment",
            status=status_th,
            max_points=4.0,
            points=_score_from_status(status_th, 4.0),
            evidence={"thresholds_found": bool(th), "saved_to": (th or {}).get("saved_to") if isinstance(th, dict) else None},
            fix_hint="Run threshold search: toxicity-agent threshold-search --config configs/train_final.yaml" if status_th != "PASS" else None,
        )
    )

    # Model versioning required under deployment considerations
    status_meta: Status = "PASS" if isinstance(meta, dict) and meta.get("run_id") and meta.get("model_name") else "FAIL"
    items.append(
        ChecklistItem(
            id="DEPLOY_VERSIONING",
            title="Model versioning metadata exists (model_metadata.json)",
            rubric_ref="I.6 (model versioning)",
            status=status_meta,
            max_points=4.0,
            points=_score_from_status(status_meta, 4.0),
            evidence={"model_metadata_found": bool(meta), "run_id": (meta or {}).get("run_id") if isinstance(meta, dict) else None},
            fix_hint="Ensure training writes model_metadata.json to the model directory (and copies to latest/)." if status_meta != "PASS" else None,
        )
    )

    # Deployment code existence
    api_main = repo_root / "src" / "toxicity_agent" / "api" / "main.py"
    dockerfile = repo_root / "Dockerfile"
    status_deploy: Status = "PASS" if api_main.exists() and dockerfile.exists() else "FAIL"
    items.append(
        ChecklistItem(
            id="DEPLOY_PIPELINE",
            title="Deployable inference pipeline exists (FastAPI + Dockerfile)",
            rubric_ref="I.6 (deployment)",
            status=status_deploy,
            max_points=8.0,
            points=_score_from_status(status_deploy, 8.0),
            evidence={"api_main": str(api_main), "dockerfile": str(dockerfile)},
            fix_hint="Add FastAPI app and Dockerfile to make system deployable." if status_deploy != "PASS" else None,
        )
    )

    # Agentic component existence
    agent_mod = repo_root / "src" / "toxicity_agent" / "agent" / "moderation_agent.py"
    status_agent: Status = "PASS" if agent_mod.exists() else "FAIL"
    items.append(
        ChecklistItem(
            id="AGENTIC",
            title="Agentic component exists (routing + policy decisions)",
            rubric_ref="I.7 (agentic AI component)",
            status=status_agent,
            max_points=8.0,
            points=_score_from_status(status_agent, 8.0),
            evidence={"moderation_agent_py": str(agent_mod)},
            fix_hint="Implement an agent that uses model outputs + policy rules + human review escalation." if status_agent != "PASS" else None,
        )
    )

    # Monitoring module existence
    monitoring = repo_root / "src" / "toxicity_agent" / "monitoring" / "daily_report.py"
    status_monitor: Status = "PASS" if monitoring.exists() else "FAIL"
    items.append(
        ChecklistItem(
            id="MONITORING",
            title="Monitoring & continual learning plan + code stubs exist",
            rubric_ref="I.8 (continual learning & monitoring)",
            status=status_monitor,
            max_points=6.0,
            points=_score_from_status(status_monitor, 6.0),
            evidence={"monitoring_module": str(monitoring)},
            fix_hint="Add monitoring aggregation and a continual learning plan." if status_monitor != "PASS" else None,
        )
    )

    # Project plan (teamwork simulation)
    status_plan: Status = "PASS" if (repo_root / "docs" / "project_plan.md").exists() else "FAIL"
    items.append(
        ChecklistItem(
            id="PROJECT_PLAN",
            title="Project plan / timeline exists",
            rubric_ref="I.10 (project management & teamwork)",
            status=status_plan,
            max_points=2.0,
            points=_score_from_status(status_plan, 2.0),
            evidence={"path": str(repo_root / "docs" / "project_plan.md")},
            fix_hint="Add project plan/timeline (docs/project_plan.md)." if status_plan != "PASS" else None,
        )
    )

    # -------------------------
    # Aggregate score
    # -------------------------
    total_max = float(sum(i.max_points for i in items))
    total_points = float(sum(i.points for i in items))

    # Determine overall status: FAIL if any critical FAIL
    critical_ids = {"SUB_REPORT", "SUB_SLIDES", "REPO_STRUCTURE", "DOCS_CORE", "ML_TUNING", "ML_EVAL_BASELINES", "ML_ERROR_ANALYSIS", "DEPLOY_PIPELINE", "AGENTIC"}
    any_critical_fail = any(i.status == "FAIL" and i.id in critical_ids for i in items)

    overall_status: Status
    if any_critical_fail:
        overall_status = "FAIL"
    else:
        overall_status = "PASS" if all(i.status == "PASS" for i in items if i.id in critical_ids) else "WARN"

    checklist = {
        "overall_status": overall_status,
        "score_estimate_100": round(total_points, 2),
        "max_score": round(total_max, 2),
        "items": [i.to_dict() for i in items],
        "notes": [
            "This checklist is an automated completeness checker aligned to the assignment rubric.",
            "The instructor's final grade may differ; quality of writing, clarity, and correctness still matter.",
        ],
    }
    return checklist


def save_grading_checklist(out_path: Path, checklist: Dict[str, Any]) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(checklist, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved grading checklist to {out_path}")
    return out_path
