from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from ..utils import ensure_dir, sha256_text


@dataclass
class PredictionLogger:
    path: Path
    store_raw_text: bool = False
    store_redacted_text: bool = False

    def __post_init__(self) -> None:
        ensure_dir(self.path.parent)

    def log(self, text: str, payload: Dict[str, Any], redacted_text: Optional[str] = None) -> None:
        record: Dict[str, Any] = {
            "ts_utc": datetime.utcnow().isoformat(),
            "text_sha256": sha256_text(text),
            **payload,
        }
        if self.store_raw_text:
            record["text"] = text
        if self.store_redacted_text and redacted_text is not None:
            record["text_redacted"] = redacted_text

        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
