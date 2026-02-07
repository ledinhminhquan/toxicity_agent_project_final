from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def normalize_text(text: str) -> str:
    return " ".join((text or "").split())


def to_label_vector(example: Dict, label_fields: Sequence[str]) -> List[float]:
    vec: List[float] = []
    for lf in label_fields:
        val = example.get(lf, 0)
        try:
            vec.append(float(val))
        except Exception:
            vec.append(0.0)
    return vec


def is_all_zero(vec: Sequence[float]) -> bool:
    return all(float(v) <= 0.0 for v in vec)


def downsample_negatives(
    texts: List[str],
    label_vectors: List[List[float]],
    ratio_keep: float,
    seed: int = 42,
) -> Tuple[List[str], List[List[float]]]:
    """Keep all positives, and only keep a ratio of all-zero negatives."""
    if ratio_keep >= 1.0:
        return texts, label_vectors

    rng = np.random.default_rng(seed)
    keep_texts: List[str] = []
    keep_labels: List[List[float]] = []

    neg_indices = [i for i, y in enumerate(label_vectors) if is_all_zero(y)]
    pos_indices = [i for i, y in enumerate(label_vectors) if not is_all_zero(y)]

    keep_neg = int(len(neg_indices) * ratio_keep)
    chosen_neg = set(rng.choice(neg_indices, size=keep_neg, replace=False).tolist()) if keep_neg > 0 else set()

    for i in pos_indices:
        keep_texts.append(texts[i])
        keep_labels.append(label_vectors[i])
    for i in chosen_neg:
        keep_texts.append(texts[i])
        keep_labels.append(label_vectors[i])

    # shuffle
    idx = rng.permutation(len(keep_texts)).tolist()
    keep_texts = [keep_texts[i] for i in idx]
    keep_labels = [keep_labels[i] for i in idx]
    return keep_texts, keep_labels
