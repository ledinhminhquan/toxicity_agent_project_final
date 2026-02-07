from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class LanguageDetector:
    default: str = "unknown"

    def detect(self, text: str) -> str:
        text = (text or "").strip()
        if len(text) < 3:
            return self.default
        try:
            from langdetect import detect  # type: ignore

            lang = detect(text)
            return lang or self.default
        except Exception:
            return self.default


def is_english(lang: str) -> bool:
    return lang.lower().startswith("en")
