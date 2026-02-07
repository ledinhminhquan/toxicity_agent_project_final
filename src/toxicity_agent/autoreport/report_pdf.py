from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..logging_utils import get_logger
from ..utils import resolve_paths

from .artifact_loader import ArtifactBundle
from .charts import (
    save_agent_flow_diagram,
    save_architecture_diagram,
    save_auc_per_label_chart,
    save_eval_comparison_chart,
    save_fairness_gap_chart,
    save_latency_chart,
)

logger = get_logger(__name__)


def _safe(v: Any, default: str = "N/A") -> str:
    if v is None:
        return default
    return str(v)


def _fmt_float(v: Any, digits: int = 4) -> str:
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return "N/A"


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _compute_label_prevalence(train_cfg: Dict[str, Any]) -> Tuple[List[str], List[float], int]:
    """Compute label prevalence on train split (no raw text)."""
    from ..data.dataset import load_and_prepare_dataset
    import numpy as np

    loaded = load_and_prepare_dataset(train_cfg)
    labels = loaded.label_fields
    ds_train = loaded.dataset["train"]
    y = np.array(ds_train["labels"], dtype=np.float32)
    if y.ndim != 2 or y.shape[1] != len(labels):
        return labels, [0.0] * len(labels), int(len(ds_train))
    rates = y.mean(axis=0).tolist()
    return labels, [float(r) for r in rates], int(len(ds_train))


def generate_report_pdf(
    *,
    bundle: ArtifactBundle,
    out_path: Path,
    title: str = "Hate Speech & Toxicity Detection System",
    author: str = "(fill) Student / Team",
    course: str = "NLP in Enterprise",
    include_appendix: bool = True,
) -> Path:
    """Generate a PDF report skeleton (10–15 pages) from artifacts.

    Robustness:
    - Missing artifacts will not crash generation; placeholders are inserted instead.
    """
    _ensure_dir(out_path)

    # Lazy import reportlab (so importing this module doesn't require reportlab unless used)
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        PageBreak,
        Image,
        Table,
        TableStyle,
        ListFlowable,
        ListItem,
    )

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], spaceAfter=8)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], spaceAfter=6)
    body = styles["BodyText"]

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        rightMargin=2.0 * cm,
        leftMargin=2.0 * cm,
        topMargin=2.0 * cm,
        bottomMargin=2.0 * cm,
        title=title,
        author=author,
    )

    story: List[Any] = []

    # Cover page
    story.append(Paragraph(title, h1))
    story.append(Paragraph(f"Course: {_safe(course)}", body))
    story.append(Paragraph(f"Author(s): {_safe(author)}", body))
    story.append(Paragraph(f"Generated automatically from run artifacts.", body))
    story.append(Spacer(1, 12))
    story.append(Paragraph("⚠️ Safety note: datasets may contain toxic content. This report intentionally avoids printing raw toxic text.", body))
    story.append(PageBreak())

    # Table of contents (simple, static)
    story.append(Paragraph("Table of Contents", h1))
    toc_items = [
        "1. Business Problem & Success Metrics",
        "2. System Overview & Architecture",
        "3. Data Management",
        "4. Model Development & Optimization",
        "5. Evaluation Results",
        "6. Error Analysis",
        "7. Agentic AI Component",
        "8. Deployment & Latency",
        "9. Monitoring & Continual Learning",
        "10. Privacy, Robustness, Ethics & Fairness",
        "11. Project Plan & Teamwork Reflection",
        "12. Conclusion & Future Work",
    ]
    story.append(ListFlowable([ListItem(Paragraph(x, body)) for x in toc_items], bulletType="1"))
    story.append(PageBreak())

    # 1) Business Problem
    story.append(Paragraph("1. Business Problem & Success Metrics", h1))
    story.append(Paragraph(
        "<b>Business context:</b> Online platforms, communities, and customer support channels face harmful toxic content, harassment, and hate speech. Manual moderation is expensive and slow. This project builds an NLP-based moderation system that automatically flags or blocks toxic content and routes uncertain cases to human review.",
        body,
    ))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<b>Stakeholders:</b> Trust & Safety team, Community managers, Customer support, Product owners.",
        body,
    ))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<b>Why NLP:</b> Toxicity appears in free-form text; keyword filters are brittle. NLP enables robust multi-label classification (toxicity types) and policy-driven actions.",
        body,
    ))
    story.append(Spacer(1, 8))

    # Success metrics from artifacts
    eval_res = bundle.eval_results or {}
    bench = bundle.benchmark or {}
    fin = eval_res.get("finetuned_transformer", {}) if isinstance(eval_res, dict) else {}

    tech_metrics = [
        ["Metric", "Value"],
        ["F1-micro (fine-tuned)", _fmt_float(fin.get("f1_micro"))],
        ["F1-macro (fine-tuned)", _fmt_float(fin.get("f1_macro"))],
        ["AUC-macro (fine-tuned)", _fmt_float(fin.get("auc_macro"))],
    ]
    # Latency metrics
    stats = bench.get("stats", {}) if isinstance(bench, dict) else {}
    tech_metrics.append(["Latency p50 (ms)", _fmt_float(stats.get("p50_ms"), 2)])
    tech_metrics.append(["Latency p95 (ms)", _fmt_float(stats.get("p95_ms"), 2)])
    tech_metrics.append(["Latency p99 (ms)", _fmt_float(stats.get("p99_ms"), 2)])

    tbl = Table(tech_metrics, hAlign="LEFT", colWidths=[7 * cm, 7 * cm])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    )
    story.append(Paragraph("Technical success metrics (auto-filled if available)", h2))
    story.append(tbl)
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "<b>Business metrics (fill):</b> moderation time saved, reduction in harmful exposure, decrease in manual review load, improved community retention. (These depend on your deployment context.)",
        body,
    ))
    story.append(PageBreak())

    # 2) System Overview & Architecture
    story.append(Paragraph("2. System Overview & Architecture", h1))
    story.append(Paragraph(
        "The system is built as an end-to-end, deployable pipeline: a training/evaluation workflow, an inference service (FastAPI), and an agentic moderation layer that routes inputs and applies policy thresholds. Outputs are logged for monitoring and continual learning.",
        body,
    ))
    story.append(Spacer(1, 8))

    # Diagrams
    tmp_dir = out_path.parent / "_figures"
    arch_img = tmp_dir / "architecture.png"
    flow_img = tmp_dir / "agent_flow.png"
    try:
        save_architecture_diagram(arch_img)
        story.append(Paragraph("System architecture diagram", h2))
        story.append(Image(str(arch_img), width=16 * cm, height=6.5 * cm))
    except Exception as e:
        logger.warning(f"Could not generate architecture diagram: {e}")
        story.append(Paragraph("[Diagram unavailable]", body))

    story.append(Spacer(1, 10))
    try:
        save_agent_flow_diagram(flow_img)
        story.append(Paragraph("Agent flow diagram", h2))
        story.append(Image(str(flow_img), width=16 * cm, height=7.0 * cm))
    except Exception as e:
        logger.warning(f"Could not generate agent flow diagram: {e}")
        story.append(Paragraph("[Diagram unavailable]", body))

    story.append(PageBreak())

    # 3) Data Management
    story.append(Paragraph("3. Data Management", h1))
    train_cfg = bundle.train_config or {}
    ds_cfg = train_cfg.get("dataset", {}) if isinstance(train_cfg, dict) else {}
    story.append(Paragraph(f"<b>Dataset:</b> {_safe(ds_cfg.get('name'))}", body))
    story.append(Paragraph(f"<b>Text field:</b> {_safe(ds_cfg.get('text_field'))}", body))
    story.append(Paragraph(f"<b>Label fields:</b> {_safe(ds_cfg.get('label_fields'))}", body))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "<b>Preprocessing:</b> normalize whitespace, basic text cleaning, and optional PII redaction (email/phone/url) before logging or manual review.",
        body,
    ))
    story.append(Spacer(1, 8))

    # Label prevalence (computed)
    if train_cfg:
        try:
            labels, rates, n_train = _compute_label_prevalence(train_cfg)
            rows = [["Label", "Positive rate"]] + [[l, _fmt_float(r, 4)] for l, r in zip(labels, rates)]
            t = Table(rows, hAlign="LEFT", colWidths=[7 * cm, 7 * cm])
            t.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ]
                )
            )
            story.append(Paragraph(f"Train split size: {n_train}", body))
            story.append(Spacer(1, 6))
            story.append(Paragraph("Label prevalence on train split", h2))
            story.append(t)
        except Exception as e:
            logger.warning(f"Could not compute label prevalence: {e}")
            story.append(Paragraph("Label prevalence: N/A", body))
    else:
        story.append(Paragraph("Label prevalence: N/A (train config missing)", body))

    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "<b>Split strategy:</b> train/validation/test split is performed deterministically with a seed. If the dataset does not provide official splits, we create splits with configurable ratios.",
        body,
    ))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<b>Known limitations:</b> class imbalance (rare labels like threats), dataset domain mismatch vs real product traffic, and potential bias regarding identity mentions.",
        body,
    ))
    story.append(PageBreak())

    # 4) Model Development & Optimization
    story.append(Paragraph("4. Model Development & Optimization", h1))
    story.append(Paragraph(
        "We compare three approaches: a traditional baseline (TF-IDF + Logistic Regression), a strong off-the-shelf baseline (Detoxify), and a fine-tuned transformer model. We perform basic hyperparameter tuning and choose a production-friendly trade-off between accuracy and latency.",
        body,
    ))
    story.append(Spacer(1, 8))

    meta = bundle.model_metadata or {}
    if meta:
        rows = [
            ["Field", "Value"],
            ["Backbone", _safe(meta.get("model_name"))],
            ["Run ID", _safe(meta.get("run_id"))],
            ["Created", _safe(meta.get("created_at_utc"))],
            ["Git commit", _safe(meta.get("git_commit"))],
            ["Precision", _safe((meta.get("environment") or {}).get("bf16_supported"), default="N/A")],
        ]
        t = Table(rows, hAlign="LEFT", colWidths=[6 * cm, 8 * cm])
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ]
            )
        )
        story.append(Paragraph("Model metadata (versioning)", h2))
        story.append(t)
        story.append(Spacer(1, 8))
    else:
        story.append(Paragraph("Model metadata not found (train may not have been run).", body))

    # Tuning summary
    tune = bundle.tune_results or {}
    if tune and isinstance(tune, dict) and tune.get("best"):
        best = tune.get("best", {})
        params = best.get("params", {})
        story.append(Paragraph("Hyperparameter tuning", h2))
        story.append(Paragraph(f"Best params: {json.dumps(params)}", body))
        story.append(Spacer(1, 6))
    else:
        story.append(Paragraph("Hyperparameter tuning: not available (skipped or not run).", body))

    story.append(PageBreak())

    # 5) Evaluation Results
    story.append(Paragraph("5. Evaluation Results", h1))
    story.append(Paragraph(
        "We evaluate using multi-label metrics: F1 (micro/macro) and ROC-AUC (macro and per-label). We also compare against baselines.",
        body,
    ))
    story.append(Spacer(1, 8))

    # table
    if eval_res:
        def row(name: str, obj: Dict[str, Any]) -> List[str]:
            return [
                name,
                _fmt_float(obj.get("f1_micro")),
                _fmt_float(obj.get("f1_macro")),
                _fmt_float(obj.get("auc_macro")),
            ]
        rows = [
            ["Model", "F1-micro", "F1-macro", "AUC-macro"],
            row("TF-IDF + LR", eval_res.get("baseline_tfidf_lr", {})),
            row("Detoxify (unbiased)", eval_res.get("baseline_detoxify_unbiased", {})),
            row("Fine-tuned transformer", eval_res.get("finetuned_transformer", {})),
        ]
        t = Table(rows, hAlign="LEFT", colWidths=[6 * cm, 3 * cm, 3 * cm, 3 * cm])
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ]
            )
        )
        story.append(t)
    else:
        story.append(Paragraph("Evaluation results not found. Run: toxicity-agent eval ...", body))

    story.append(Spacer(1, 10))

    # charts
    try:
        eval_cmp = tmp_dir / "eval_comparison.png"
        p = save_eval_comparison_chart(eval_res, eval_cmp) if eval_res else None
        if p:
            story.append(Paragraph("F1 comparison chart", h2))
            story.append(Image(str(p), width=15.5 * cm, height=7 * cm))
            story.append(Spacer(1, 8))
    except Exception as e:
        logger.warning(f"Could not create eval chart: {e}")

    try:
        auc_img = tmp_dir / "auc_per_label.png"
        p = save_auc_per_label_chart(eval_res, auc_img) if eval_res else None
        if p:
            story.append(Paragraph("Per-label ROC-AUC (fine-tuned)", h2))
            story.append(Image(str(p), width=16 * cm, height=7 * cm))
    except Exception as e:
        logger.warning(f"Could not create AUC per label chart: {e}")

    story.append(PageBreak())

    # 6) Error Analysis
    story.append(Paragraph("6. Error Analysis", h1))
    err = bundle.error_analysis or {}
    if err:
        story.append(Paragraph(
            "This section summarizes model failure patterns without exposing raw text. We report confusion counts per label, plus aggregate feature patterns for false positives/negatives.",
            body,
        ))
        story.append(Spacer(1, 8))

        # Confusion counts table (for a few labels)
        conf = err.get("confusion_per_label", {})
        if isinstance(conf, dict) and conf:
            rows = [["Label", "TP", "FP", "FN", "TN"]]
            for lf, c in conf.items():
                if not isinstance(c, dict):
                    continue
                rows.append([lf, _safe(c.get("tp")), _safe(c.get("fp")), _safe(c.get("fn")), _safe(c.get("tn"))])
            t = Table(rows, hAlign="LEFT", colWidths=[5 * cm, 2 * cm, 2 * cm, 2 * cm, 2 * cm])
            t.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ]
                )
            )
            story.append(Paragraph("Confusion counts per label", h2))
            story.append(t)
            story.append(Spacer(1, 10))

        # Top cases (hashed)
        top = err.get("top_error_cases", [])
        if isinstance(top, list) and top:
            rows = [["sha256(text)", "max_abs_error", "len_chars", "uppercase_ratio"]]
            for c in top[:10]:
                if not isinstance(c, dict):
                    continue
                rows.append([
                    _safe(c.get("sha256"))[:16] + "…",
                    _fmt_float(c.get("max_abs_error"), 3),
                    _safe(c.get("length_chars")),
                    _fmt_float(c.get("uppercase_ratio"), 3),
                ])
            t2 = Table(rows, hAlign="LEFT", colWidths=[6 * cm, 3 * cm, 3 * cm, 3 * cm])
            t2.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ]
                )
            )
            story.append(Paragraph("Top error cases (privacy-preserving identifiers only)", h2))
            story.append(t2)

        story.append(Spacer(1, 10))
        story.append(Paragraph(
            "<b>Action items:</b> (1) add in-domain training data, (2) adjust thresholds for rare labels, (3) add active learning loop using human review queue.",
            body,
        ))
    else:
        story.append(Paragraph("Error analysis not found. Run: toxicity-agent error-analysis ...", body))

    story.append(PageBreak())

    # 7) Agentic AI Component
    story.append(Paragraph("7. Agentic AI Component", h1))
    story.append(Paragraph(
        "The moderation agent performs multi-step decision-making: it detects language, routes inputs to different models (fast vs strong) based on risk/uncertainty, applies policy thresholds, logs outcomes, and optionally escalates to human review.",
        body,
    ))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "This fulfills the 'agentic' requirement via: routing, intermediate scoring, decision policy, and tool usage (models + policy store + queues).", body
    ))
    story.append(PageBreak())

    # 8) Deployment & Latency
    story.append(Paragraph("8. Deployment & Latency", h1))
    story.append(Paragraph(
        "Deployment format: REST API (FastAPI) providing /health and /v1/moderate endpoints. The agent runs inside the API server and returns structured JSON decisions.",
        body,
    ))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Latency is measured end-to-end for the moderation agent (including model calls). This is critical for production readiness.",
        body,
    ))
    story.append(Spacer(1, 8))

    if bench:
        stats = bench.get("stats", {})
        rows = [
            ["Latency metric", "Value (ms)"],
            ["p50", _fmt_float(stats.get("p50_ms"), 2)],
            ["p95", _fmt_float(stats.get("p95_ms"), 2)],
            ["p99", _fmt_float(stats.get("p99_ms"), 2)],
            ["mean", _fmt_float(stats.get("mean_ms"), 2)],
        ]
        t = Table(rows, hAlign="LEFT", colWidths=[7 * cm, 7 * cm])
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ]
            )
        )
        story.append(t)
        story.append(Spacer(1, 10))
        try:
            lat_img = tmp_dir / "latency.png"
            p = save_latency_chart(bench, lat_img)
            if p:
                story.append(Image(str(p), width=12 * cm, height=7 * cm))
        except Exception as e:
            logger.warning(f"Could not generate latency chart: {e}")
    else:
        story.append(Paragraph("Latency benchmark not found. Run: toxicity-agent benchmark ...", body))

    story.append(PageBreak())

    # 9) Monitoring & Continual Learning
    story.append(Paragraph("9. Monitoring & Continual Learning", h1))
    story.append(Paragraph(
        "The system logs prediction events as JSONL. A monitoring job aggregates these logs into daily metrics (volume, toxicity rate, action distribution). Over time, drift can be detected by changes in score distributions, label prevalence, and complaint rate.",
        body,
    ))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "<b>Continual learning strategy:</b> collect new data from the human review queue; periodically retrain; compare against baselines; deploy using model versioning and rollback.",
        body,
    ))
    story.append(PageBreak())

    # 10) Privacy, Robustness, Ethics & Fairness
    story.append(Paragraph("10. Privacy, Robustness, Ethics & Fairness", h1))
    story.append(Paragraph(
        "<b>Privacy:</b> avoid storing raw toxic text in reports; redact basic PII in logs; store hashed identifiers for error analysis. Do not commit raw datasets to the repository.",
        body,
    ))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<b>Robustness:</b> test noisy inputs; handle out-of-domain language; provide a safe fallback and human review escalation.",
        body,
    ))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<b>Ethics:</b> the system can reduce harm but may also over-block legitimate speech. Use transparent policy thresholds, provide appeal/human review, and monitor for disparate impact.",
        body,
    ))
    story.append(Spacer(1, 10))

    fair = bundle.fairness or {}
    if fair:
        story.append(Paragraph("Fairness slice evaluation (identity mentions)", h2))
        story.append(Paragraph(
            "We compute performance on slices defined by benign identity-term mentions (heuristic matching). We report gaps vs a no-identity-mention reference slice.",
            body,
        ))
        story.append(Spacer(1, 6))
        try:
            fair_img = tmp_dir / "fairness_gap_toxic.png"
            p = save_fairness_gap_chart(fair, fair_img, label="toxic")
            if p:
                story.append(Image(str(p), width=16 * cm, height=6.5 * cm))
        except Exception as e:
            logger.warning(f"Could not generate fairness chart: {e}")
        story.append(Spacer(1, 8))
        story.append(Paragraph(
            "<b>Interpretation:</b> large positive FPR gaps may indicate over-flagging in identity-mention contexts; large negative TPR gaps may indicate under-detection. This heuristic must be complemented with domain-specific review.",
            body,
        ))
    else:
        story.append(Paragraph("Fairness report not found. Run: toxicity-agent fairness ...", body))

    story.append(PageBreak())

    # 11) Project Plan & Teamwork Reflection
    story.append(Paragraph("11. Project Plan & Teamwork Reflection", h1))
    story.append(Paragraph(
        "This project was planned as a mini production system. Even for a solo project, we simulate team roles: data engineer (data pipeline), ML engineer (training), backend engineer (API), and PM (docs/report).",
        body,
    ))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<b>Timeline (suggested):</b> Week 1: setup + baselines; Week 2: fine-tune + eval; Week 3: agent + API + monitoring; Week 4: error analysis + fairness + report/slides.",
        body,
    ))
    story.append(PageBreak())

    # 12) Conclusion & Future Work
    story.append(Paragraph("12. Conclusion & Future Work", h1))
    story.append(Paragraph(
        "We delivered an end-to-end toxicity moderation system with training, evaluation, agentic decision-making, deployment, monitoring, and responsible AI considerations. Next steps include adding in-domain data, improving threshold calibration, and expanding multilingual coverage.",
        body,
    ))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "<b>Future work ideas:</b> active learning loop, drift dashboards, per-language thresholds, robust adversarial testing, and human-in-the-loop feedback tooling.",
        body,
    ))

    if include_appendix:
        story.append(PageBreak())
        story.append(Paragraph("Appendix: Auto-filled artifact snapshots", h1))
        story.append(Paragraph(
            "This appendix includes compact JSON snippets to support reproducibility. (No raw text.)",
            body,
        ))
        story.append(Spacer(1, 8))
        # Keep appendix short to avoid huge PDFs
        snap = {
            "eval_results_present": bool(bundle.eval_results),
            "benchmark_present": bool(bundle.benchmark),
            "error_analysis_present": bool(bundle.error_analysis),
            "fairness_present": bool(bundle.fairness),
            "model_metadata_present": bool(bundle.model_metadata),
        }
        story.append(Paragraph(json.dumps(snap, indent=2), body))

    doc.build(story)
    logger.info(f"Generated report PDF at {out_path}")
    return out_path
