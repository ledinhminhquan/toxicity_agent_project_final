from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from ..utils import ensure_dir


@dataclass
class HumanReviewQueue:
    path: Path

    def __post_init__(self) -> None:
        ensure_dir(self.path.parent)

    def enqueue(self, item: Dict) -> None:
        record = {
            "ts_utc": datetime.utcnow().isoformat(),
            **item,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
