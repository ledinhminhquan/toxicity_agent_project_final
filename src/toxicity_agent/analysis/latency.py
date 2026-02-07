from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

import numpy as np

from ..logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class LatencyStats:
    n: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "mean_ms": self.mean_ms,
        }


def benchmark_callable(
    fn: Callable[[str], Any],
    *,
    texts: Sequence[str],
    warmup: int = 10,
) -> LatencyStats:
    """Benchmark latency for a callable that accepts a single text and returns anything.

    This is intentionally simple and stable on Colab/CPU.
    """
    texts = list(texts)
    n = len(texts)
    if n == 0:
        return LatencyStats(n=0, p50_ms=0.0, p95_ms=0.0, p99_ms=0.0, mean_ms=0.0)

    # Warmup
    for t in texts[: min(warmup, n)]:
        _ = fn(t)

    lat_ms: List[float] = []
    for t in texts:
        t0 = time.perf_counter()
        _ = fn(t)
        lat_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(lat_ms, dtype=np.float64)
    return LatencyStats(
        n=n,
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        mean_ms=float(arr.mean()),
    )


def save_latency_report(out_path: Path, report: Dict[str, Any]) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved latency report to {out_path}")
    return out_path
