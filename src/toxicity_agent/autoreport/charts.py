from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_eval_comparison_chart(eval_results: Dict[str, Any], out_path: Path) -> Optional[Path]:
    """Bar chart comparing F1_micro across models."""
    if not eval_results:
        return None

    models = [
        ("TF-IDF + LR", eval_results.get("baseline_tfidf_lr", {})),
        ("Detoxify (unbiased)", eval_results.get("baseline_detoxify_unbiased", {})),
        ("Fine-tuned", eval_results.get("finetuned_transformer", {})),
    ]

    names: List[str] = []
    vals: List[float] = []
    for name, obj in models:
        v = obj.get("f1_micro")
        if v is None:
            continue
        names.append(name)
        vals.append(float(v))

    if not names:
        return None

    plt.figure(figsize=(8, 4))
    plt.bar(names, vals)
    plt.ylim(0.0, 1.0)
    plt.title("F1-micro comparison")
    plt.ylabel("F1-micro")
    plt.xticks(rotation=20, ha="right")
    _ensure_parent(out_path)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def save_auc_per_label_chart(eval_results: Dict[str, Any], out_path: Path) -> Optional[Path]:
    fin = eval_results.get("finetuned_transformer") if isinstance(eval_results, dict) else None
    if not isinstance(fin, dict):
        return None
    auc = fin.get("auc_per_label")
    if not isinstance(auc, dict) or not auc:
        return None

    labels = list(auc.keys())
    vals = [0.0 if auc[l] is None else float(auc[l]) for l in labels]

    plt.figure(figsize=(9, 4))
    plt.bar(labels, vals)
    plt.ylim(0.0, 1.0)
    plt.title("Fine-tuned model: ROC-AUC per label")
    plt.ylabel("ROC-AUC")
    plt.xticks(rotation=25, ha="right")
    _ensure_parent(out_path)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def save_latency_chart(benchmark: Dict[str, Any], out_path: Path) -> Optional[Path]:
    if not isinstance(benchmark, dict):
        return None
    stats = benchmark.get("stats")
    if not isinstance(stats, dict):
        return None
    keys = ["p50_ms", "p95_ms", "p99_ms"]
    vals = [float(stats.get(k, 0.0) or 0.0) for k in keys]

    plt.figure(figsize=(6, 3.5))
    plt.bar(["p50", "p95", "p99"], vals)
    plt.title("Agent latency (ms)")
    plt.ylabel("milliseconds")
    _ensure_parent(out_path)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def save_fairness_gap_chart(fairness: Dict[str, Any], out_path: Path, label: str = "toxic") -> Optional[Path]:
    """Plot TPR/FPR gap vs no-identity slice for one label."""
    if not isinstance(fairness, dict):
        return None
    gaps = fairness.get("gaps_vs_no_identity")
    if not isinstance(gaps, dict) or not gaps:
        return None

    groups = []
    tpr = []
    fpr = []
    for slice_name, g in gaps.items():
        tg = g.get("tpr_gap", {})
        fg = g.get("fpr_gap", {})
        if not isinstance(tg, dict) or not isinstance(fg, dict):
            continue
        if label not in tg or label not in fg:
            continue
        groups.append(slice_name)
        tpr.append(float(tg[label]))
        fpr.append(float(fg[label]))

    if not groups:
        return None

    # Two bars per group: tpr gap and fpr gap
    import numpy as np

    x = np.arange(len(groups))
    width = 0.35

    plt.figure(figsize=(10, 4))
    plt.bar(x - width / 2, tpr, width, label="TPR gap")
    plt.bar(x + width / 2, fpr, width, label="FPR gap")
    plt.axhline(0.0)
    plt.title(f"Fairness gaps vs no-identity (label={label})")
    plt.xticks(x, groups, rotation=25, ha="right")
    plt.legend()
    _ensure_parent(out_path)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def save_architecture_diagram(out_path: Path) -> Path:
    """Create a simple architecture diagram using matplotlib shapes.

    This avoids external dependencies like graphviz.
    """
    from matplotlib.patches import FancyBboxPatch, ArrowStyle
    import matplotlib.patches as patches

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.axis("off")

    def box(x, y, w, h, text):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", linewidth=1)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

    # boxes
    box(0.05, 0.55, 0.18, 0.3, "Input text\n(API/Batch)")
    box(0.30, 0.55, 0.18, 0.3, "Moderation Agent\n(language + routing)")
    box(0.55, 0.70, 0.18, 0.15, "Fast model\nDetoxify-small")
    box(0.55, 0.50, 0.18, 0.15, "Strong model\nFine-tuned HF")
    box(0.80, 0.55, 0.18, 0.3, "Policy Decision\nALLOW/WARN/REVIEW/BLOCK")
    box(0.30, 0.10, 0.18, 0.25, "Logs & Monitoring\n(JSONL + report)")
    box(0.55, 0.10, 0.18, 0.25, "Human review queue\n(optional)")
    box(0.80, 0.10, 0.18, 0.25, "Continual learning\n(retrain plan)")

    def arrow(x1, y1, x2, y2):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", linewidth=1),
        )

    # arrows
    arrow(0.23, 0.70, 0.30, 0.70)
    arrow(0.48, 0.70, 0.55, 0.78)
    arrow(0.48, 0.70, 0.55, 0.58)
    arrow(0.73, 0.78, 0.80, 0.70)
    arrow(0.73, 0.58, 0.80, 0.70)
    arrow(0.39, 0.55, 0.39, 0.35)
    arrow(0.64, 0.55, 0.64, 0.35)
    arrow(0.89, 0.55, 0.89, 0.35)
    arrow(0.48, 0.23, 0.55, 0.23)
    arrow(0.73, 0.23, 0.80, 0.23)

    _ensure_parent(out_path)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def save_agent_flow_diagram(out_path: Path) -> Path:
    """Create a simple agent flow diagram."""
    from matplotlib.patches import FancyBboxPatch

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.axis("off")

    steps = [
        "1) Detect language",
        "2) Fast toxicity scoring\n(Detoxify-small)",
        "3) If borderline/high-risk:\n run second-pass model",
        "4) Apply policy thresholds",
        "5) Output action + optional human review",
    ]

    y = 0.85
    for i, s in enumerate(steps):
        p = FancyBboxPatch((0.1, y - 0.12), 0.8, 0.1, boxstyle="round,pad=0.02", linewidth=1)
        ax.add_patch(p)
        ax.text(0.5, y - 0.07, s, ha="center", va="center", fontsize=11)
        if i < len(steps) - 1:
            ax.annotate("", xy=(0.5, y - 0.12), xytext=(0.5, y - 0.16), arrowprops=dict(arrowstyle="->"))
        y -= 0.16

    _ensure_parent(out_path)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path
