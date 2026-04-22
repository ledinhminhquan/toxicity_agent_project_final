"""Vietnamese ViHSD dataset adapter.

Loads `visolex/ViHSD` from the Hugging Face Hub and maps its 3-class label
(``CLEAN`` / ``OFFENSIVE`` / ``HATE``) into a 2-column multi-label matrix
compatible with the project's existing multi-label training pipeline.

Label mapping (sidecar taxonomy):

    CLEAN      -> offensive=0, hate=0
    OFFENSIVE  -> offensive=1, hate=0
    HATE       -> offensive=0, hate=1

The adapter is intentionally self-contained so it can be swapped in without
touching the English Jigsaw pipeline. It is imported lazily (e.g. by the VI
notebook and by a future CLI hook) to keep unit tests that only exercise the
English path from pulling the HuggingFace Datasets dependency at import time.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..logging_utils import get_logger

logger = get_logger(__name__)


VI_LABEL_FIELDS: List[str] = ["offensive", "hate"]

_SPLIT_ALIASES: Dict[str, str] = {
    "train": "train",
    "training": "train",
    "val": "validation",
    "valid": "validation",
    "dev": "validation",
    "validation": "validation",
    "test": "test",
    "testing": "test",
}

_TEXT_CANDIDATES: Tuple[str, ...] = ("text", "free_text", "comment", "comment_text")
_LABEL_CANDIDATES: Tuple[str, ...] = ("label", "label_id")
_SPLIT_FIELD: str = "type"


def _canonical_split_name(value: Any) -> str:
    s = str(value).strip().lower()
    return _SPLIT_ALIASES.get(s, s)


def _candidate_column(columns: List[str], candidates: Tuple[str, ...]) -> str:
    for c in candidates:
        if c in columns:
            return c
    raise KeyError(f"Could not find any of {list(candidates)} in columns={columns}")


def _parse_label_to_class_id(value: Any) -> int:
    """Normalize a ViHSD label (str or int) into 0 (CLEAN) / 1 (OFFENSIVE) / 2 (HATE)."""
    if value is None:
        return 0
    if isinstance(value, str):
        s = value.strip().upper()
        if s in {"CLEAN", "NON_TOXIC", "SAFE"}:
            return 0
        if s in {"OFFENSIVE", "TOXIC"}:
            return 1
        if s in {"HATE", "HATE_SPEECH"}:
            return 2
        try:
            i = int(float(value))
        except Exception:
            return 0
        return {0: 0, 1: 1, 2: 2}.get(i, 0)
    try:
        i = int(value)
    except Exception:
        return 0
    return {0: 0, 1: 1, 2: 2}.get(i, 0)


def _class_id_to_multi_hot(class_id: int) -> List[float]:
    if class_id == 1:
        return [1.0, 0.0]  # offensive
    if class_id == 2:
        return [0.0, 1.0]  # hate
    return [0.0, 0.0]      # clean


def _normalize_whitespace(text: str) -> str:
    return " ".join((text or "").split())


def load_vihsd(
    dataset_name: str = "visolex/ViHSD",
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
    seed: int = 42,
) -> Tuple[Any, Dict[str, Any]]:
    """Load and normalize the ViHSD dataset.

    Returns a ``(DatasetDict, summary)`` pair where the DatasetDict has
    ``train`` / ``validation`` / ``test`` splits and the summary records which
    text and label columns were actually picked up (ViHSD has shown minor
    field-name inconsistencies across revisions).

    The returned rows have these columns:
        - ``id``: short hash of the text (stable id)
        - ``text``: whitespace-normalized text
        - ``labels``: list[float] of length 2, order matches VI_LABEL_FIELDS
        - ``class_id``: int 0/1/2
        - ``class_name``: str
        - ``split``: canonical split name
        - ``char_len``: int
    """
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover - Colab-only path
        raise ImportError(
            "datasets library is required to load ViHSD. "
            "Install it with `pip install datasets`."
        ) from e

    import hashlib

    def _short_hash(text: str, n: int = 12) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]

    raw = load_dataset(dataset_name, split="train")
    cols = list(raw.column_names)
    text_col = _candidate_column(cols, _TEXT_CANDIDATES)

    label_col = None
    for c in _LABEL_CANDIDATES:
        if c in cols:
            label_col = c
            break
    if label_col is None:
        raise KeyError(f"Could not find label column in {cols}")

    split_col = _SPLIT_FIELD if _SPLIT_FIELD in cols else None

    def _map_fn(ex: Dict[str, Any]) -> Dict[str, Any]:
        text = _normalize_whitespace(str(ex.get(text_col, "")))
        class_id = _parse_label_to_class_id(ex.get(label_col))
        split_name = _canonical_split_name(ex.get(split_col, "train")) if split_col else "train"
        return {
            "id": _short_hash(text),
            "text": text,
            "labels": _class_id_to_multi_hot(class_id),
            "class_id": int(class_id),
            "class_name": {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}.get(int(class_id), "CLEAN"),
            "split": split_name,
            "char_len": len(text),
        }

    ds = raw.map(_map_fn, remove_columns=cols)

    if split_col is not None:
        train = ds.filter(lambda ex: ex["split"] == "train")
        val = ds.filter(lambda ex: ex["split"] == "validation")
        test = ds.filter(lambda ex: ex["split"] == "test")
        if len(train) == 0:
            raise RuntimeError("No train rows found after ViHSD split normalization.")
        if len(val) == 0 or len(test) == 0:
            logger.warning("Missing validation/test after split normalization -> falling back to random split.")
            _use_internal_split = False
        else:
            _use_internal_split = True
    else:
        _use_internal_split = False

    if not _use_internal_split:
        tmp = ds.train_test_split(test_size=0.20, seed=seed)
        train = tmp["train"]
        rest = tmp["test"].train_test_split(test_size=0.50, seed=seed)
        val = rest["train"]
        test = rest["test"]

    if max_train_samples is not None:
        train = train.select(range(min(len(train), int(max_train_samples))))
    if max_eval_samples is not None:
        val = val.select(range(min(len(val), int(max_eval_samples))))
        test = test.select(range(min(len(test), int(max_eval_samples))))

    from datasets import DatasetDict  # type: ignore

    dsd = DatasetDict(train=train, validation=val, test=test)

    def _dist(split_ds) -> Dict[str, Any]:
        import numpy as np

        class_ids = np.asarray(split_ds["class_id"], dtype=np.int64)
        char_lens = np.asarray(split_ds["char_len"], dtype=np.int64)
        return {
            "n_rows": int(len(split_ds)),
            "clean": int((class_ids == 0).sum()),
            "offensive": int((class_ids == 1).sum()),
            "hate": int((class_ids == 2).sum()),
            "avg_char_len": float(char_lens.mean()) if len(char_lens) else 0.0,
            "p95_char_len": int(np.percentile(char_lens, 95)) if len(char_lens) else 0,
        }

    summary = {
        "dataset_name": dataset_name,
        "columns_detected": cols,
        "text_column_used": text_col,
        "label_column_used": label_col,
        "split_column_used": split_col,
        "label_fields": list(VI_LABEL_FIELDS),
        "splits": {k: _dist(v) for k, v in dsd.items()},
    }
    return dsd, summary


# Public re-exports for easy import
__all__ = [
    "VI_LABEL_FIELDS",
    "load_vihsd",
]
