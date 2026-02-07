from __future__ import annotations

from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field
from pydantic import ConfigDict


class ModerateRequest(BaseModel):
    text: Optional[str] = Field(default=None, description="Single input text")
    texts: Optional[List[str]] = Field(default=None, description="Batch input texts")

    def get_texts(self) -> List[str]:
        if self.texts is not None:
            return self.texts
        if self.text is None:
            return []
        return [self.text]


class ModerationResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    action: str
    overall_score: float
    label_scores: Dict[str, float]
    model_used: str
    language: str
    policy_message: str
    explanation: str
    request_human_review: bool = False


class ModerateResponse(BaseModel):
    results: List[ModerationResult]