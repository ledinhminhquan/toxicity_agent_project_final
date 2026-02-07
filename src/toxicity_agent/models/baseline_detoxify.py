from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Union

import numpy as np


def _batched(seq: Sequence[str], batch_size: int) -> Sequence[Sequence[str]]:
    if batch_size <= 0:
        batch_size = 32
    return [seq[i : i + batch_size] for i in range(0, len(seq), batch_size)]


@dataclass
class DetoxifyPredictor:
    """Light wrapper around Detoxify for consistent multi-label probabilities.

    Notes
    -----
    - Detoxify internally downloads model weights on first use (requires internet).
    - Some Detoxify model variants use `identity_attack` instead of `identity_hate`.
      We support mapping via `label_map`.
    """

    model_type: str
    device: str = "cpu"

    def __post_init__(self) -> None:
        from detoxify import Detoxify  # local import (heavy dependency)

        self._model = Detoxify(self.model_type, device=self.device)

    def predict_proba(self, texts: Union[str, Sequence[str]]) -> Dict[str, List[float]]:
        # Detoxify accepts a string or list of strings.
        if isinstance(texts, str):
            texts_list = [texts]
        else:
            texts_list = list(texts)

        out = self._model.predict(texts_list)
        # Ensure python floats
        return {k: [float(x) for x in v] for k, v in out.items()}

    def predict_proba_matrix(
        self,
        texts: Sequence[str],
        label_order: Sequence[str],
        label_map: Dict[str, str] | None = None,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Return NxL matrix in the desired label order (batched for stability).

        Parameters
        ----------
        texts:
            Sequence of input strings.
        label_order:
            Desired output label order.
        label_map:
            Optional mapping desired_label -> detoxify_label_key.
            Example: {'identity_hate': 'identity_attack'}
        batch_size:
            Batch size for Detoxify inference.

        Returns
        -------
        np.ndarray
            Shape (N, L) float32.
        """
        n = len(texts)
        if n == 0:
            return np.zeros((0, len(label_order)), dtype=np.float32)

        mats: List[np.ndarray] = []
        for chunk in _batched(list(texts), batch_size=batch_size):
            preds = self.predict_proba(chunk)
            mat = np.zeros((len(chunk), len(label_order)), dtype=np.float32)

            for j, desired_label in enumerate(label_order):
                key = desired_label
                if label_map and desired_label in label_map:
                    key = label_map[desired_label]
                if key not in preds:
                    continue
                mat[:, j] = np.array(preds[key], dtype=np.float32)

            mats.append(mat)

        return np.vstack(mats)
