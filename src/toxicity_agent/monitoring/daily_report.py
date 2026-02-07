from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


@dataclass
class DailyReport:
    n: int
    action_counts: Dict[str, int]
    language_counts: Dict[str, int]
    avg_overall_score: float
    avg_label_scores: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "action_counts": self.action_counts,
            "language_counts": self.language_counts,
            "avg_overall_score": self.avg_overall_score,
            "avg_label_scores": self.avg_label_scores,
        }


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def generate_report(predictions_log: Path) -> DailyReport:
    action_counter = Counter()
    lang_counter = Counter()
    overall_scores: List[float] = []
    label_sum = defaultdict(float)
    label_count = defaultdict(int)

    n = 0
    for rec in _read_jsonl(predictions_log):
        n += 1
        action_counter.update([rec.get("action", "UNKNOWN")])
        lang_counter.update([rec.get("language", "unknown")])
        try:
            overall_scores.append(float(rec.get("overall_score", 0.0)))
        except Exception:
            pass
        label_scores = rec.get("label_scores", {}) or {}
        for k, v in label_scores.items():
            try:
                label_sum[k] += float(v)
                label_count[k] += 1
            except Exception:
                pass

    avg_overall = float(np.mean(overall_scores)) if overall_scores else 0.0
    avg_labels = {k: (label_sum[k] / max(label_count[k], 1)) for k in label_sum.keys()}

    return DailyReport(
        n=n,
        action_counts=dict(action_counter),
        language_counts=dict(lang_counter),
        avg_overall_score=avg_overall,
        avg_label_scores=avg_labels,
    )
