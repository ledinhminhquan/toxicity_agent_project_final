from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _batched(seq: Sequence[str], batch_size: int) -> Sequence[Sequence[str]]:
    if batch_size <= 0:
        batch_size = 32
    return [seq[i : i + batch_size] for i in range(0, len(seq), batch_size)]


@dataclass
class HFPredictor:
    """Predictor for a fine-tuned HuggingFace multi-label classifier.

    This class performs *batched* inference to avoid OOM and keep Colab stable.
    """

    model_dir: Path
    device: str = "cpu"
    max_length: int = 256
    label_fields: List[str] | None = None

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_proba_matrix(self, texts: Sequence[str], label_order: Sequence[str], batch_size: int = 64) -> np.ndarray:
        n = len(texts)
        if n == 0:
            return np.zeros((0, len(label_order)), dtype=np.float32)

        probs_all: List[np.ndarray] = []
        for chunk in _batched(list(texts), batch_size=batch_size):
            enc = self.tokenizer(
                list(chunk),
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            logits = self.model(**enc).logits
            probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
            if probs.shape[1] != len(label_order):
                raise ValueError(f"Model outputs {probs.shape[1]} labels but expected {len(label_order)}")
            probs_all.append(probs)

        return np.vstack(probs_all)
