from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class Action(str, Enum):
    ALLOW = "ALLOW"
    WARN = "WARN"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"


@dataclass
class ModerationDecision:
    action: Action
    overall_score: float
    label_scores: Dict[str, float]
    model_used: str
    language: str
    policy_message: str
    explanation: str
    request_human_review: bool = False
    redacted_for_logs: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "action": self.action.value,
            "overall_score": self.overall_score,
            "label_scores": self.label_scores,
            "model_used": self.model_used,
            "language": self.language,
            "policy_message": self.policy_message,
            "explanation": self.explanation,
            "request_human_review": self.request_human_review,
        }
