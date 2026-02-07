from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from .actions import Action


@dataclass
class PolicyStore:
    rules: Dict[str, Any]

    @staticmethod
    def load(path: Path) -> "PolicyStore":
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return PolicyStore(rules=data or {})

    def action_message(self, action: Action) -> str:
        return str(self.rules.get("actions", {}).get(action.value, {}).get("user_message", ""))

    def label_display_name(self, label: str) -> str:
        return str(self.rules.get("labels", {}).get(label, {}).get("display_name", label))

    def explanation_top_k(self) -> int:
        return int(self.rules.get("explanations", {}).get("top_k_labels", 3))

    def explanation_include_scores(self) -> bool:
        return bool(self.rules.get("explanations", {}).get("include_scores", True))
