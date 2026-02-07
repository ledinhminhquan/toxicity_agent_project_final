from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from transformers import AutoTokenizer

from .preprocessing import normalize_text, to_label_vector

logger = logging.getLogger(__name__)


@dataclass
class LoadedData:
    dataset: DatasetDict
    label_fields: List[str]
    text_field: str
    id_field: Optional[str]
    tokenizer_name: str


def _has_labels(example: Dict[str, Any], label_fields: Sequence[str]) -> bool:
    """Drop rows with -1 labels (Kaggle test_labels convention) or missing labels."""
    for lf in label_fields:
        v = example.get(lf, None)
        if v is None:
            return False
        try:
            if float(v) < 0:
                return False
        except Exception:
            return False
    return True


def _split_has_label_columns(split: Dataset, label_fields: Sequence[str]) -> bool:
    """Check whether a dataset split actually contains the required label columns."""
    cols = set(split.column_names)
    return all(lf in cols for lf in label_fields)


def _safe_load_dataset(ds_name: str) -> DatasetDict:
    """Load dataset, handling cases where splits have incompatible schemas.

    Many HuggingFace datasets that mirror Kaggle competitions have a test split
    whose columns differ from the train split (e.g., test.csv has no label columns).
    The ``datasets`` library raises a ``CastError`` when it tries to unify the
    Arrow schemas — even when ``split="train"`` is specified, because the builder
    processes *all* CSV files before returning the requested split.

    When that happens we fall back to loading only the ``train.csv`` file via
    ``data_files``, which tells the builder to ignore other files entirely.
    """
    try:
        return load_dataset(ds_name)
    except Exception as exc:
        logger.warning(
            "load_dataset('%s') failed (%s). "
            "Falling back to loading only train.csv via data_files.",
            ds_name,
            exc.__class__.__name__,
        )
        # data_files="train.csv" tells the builder to ONLY process train.csv,
        # completely avoiding the schema mismatch with test.csv.
        try:
            ds = load_dataset(ds_name, data_files="train.csv", split="train")
            return DatasetDict({"train": ds})
        except Exception:
            # Last resort: some datasets may use different file names
            ds = load_dataset(ds_name, data_files={"train": "train.csv"}, split="train")
            return DatasetDict({"train": ds})


def _prepare_split(split: Dataset, text_field: str, label_fields: Sequence[str], id_field: Optional[str]) -> Dataset:
    def _map(ex: Dict[str, Any]) -> Dict[str, Any]:
        text = normalize_text(ex.get(text_field, ""))
        labels = to_label_vector(ex, label_fields)
        out: Dict[str, Any] = {"text": text, "labels": labels}
        if id_field and id_field in ex:
            out["id"] = ex[id_field]
        return out

    # Filter to labeled rows only
    split = split.filter(lambda ex: _has_labels(ex, label_fields))
    # Remove all original columns and replace with our normalized fields
    split = split.map(_map, remove_columns=list(split.column_names))
    return split


def _downsample_train(ds_train: Dataset, ratio: float, seed: int) -> Dataset:
    if ratio >= 1.0:
        return ds_train

    def _is_positive(ex: Dict[str, Any]) -> bool:
        labels = ex.get("labels", [])
        try:
            return any(float(v) > 0 for v in labels)
        except Exception:
            return False

    pos = ds_train.filter(_is_positive)
    neg = ds_train.filter(lambda ex: not _is_positive(ex))
    keep = int(len(neg) * ratio)
    if keep <= 0:
        return pos.shuffle(seed=seed)
    neg_keep = neg.shuffle(seed=seed).select(range(keep))
    merged = concatenate_datasets([pos, neg_keep]).shuffle(seed=seed)
    return merged


def load_and_prepare_dataset(cfg: Dict[str, Any]) -> LoadedData:
    ds_name = cfg["dataset"]["name"]
    text_field = cfg["dataset"]["text_field"]
    id_field = cfg["dataset"].get("id_field") or None
    label_fields = list(cfg["dataset"]["label_fields"])
    neg_ratio = float(cfg["dataset"].get("negative_downsample_ratio", 1.0))
    seed = int(cfg.get("project", {}).get("seed", 42))

    raw = _safe_load_dataset(ds_name)

    # Determine which splits are usable (have label columns)
    if "train" in raw:
        train_raw = raw["train"]
    else:
        first_key = list(raw.keys())[0]
        train_raw = raw[first_key]

    # Only use val/test splits if they actually contain label columns.
    # Kaggle-mirror datasets often have a test split with NO labels.
    val_raw = raw.get("validation")
    test_raw = raw.get("test")

    if val_raw is not None and not _split_has_label_columns(val_raw, label_fields):
        logger.info("Validation split exists but has no label columns; ignoring it.")
        val_raw = None
    if test_raw is not None and not _split_has_label_columns(test_raw, label_fields):
        logger.info("Test split exists but has no label columns; ignoring it.")
        test_raw = None

    # Prepare train split
    train = _prepare_split(train_raw, text_field=text_field, label_fields=label_fields, id_field=id_field)
    train = _downsample_train(train, ratio=neg_ratio, seed=seed)

    if val_raw is not None and test_raw is not None:
        val = _prepare_split(val_raw, text_field=text_field, label_fields=label_fields, id_field=id_field)
        test = _prepare_split(test_raw, text_field=text_field, label_fields=label_fields, id_field=id_field)
        # Only use external splits if they have enough labeled rows after filtering
        if len(val) > 0 and len(test) > 0:
            ds = DatasetDict(train=train, validation=val, test=test)
        else:
            logger.info("External val/test splits empty after label filtering; creating splits from train.")
            val_raw = None  # fall through to split creation below

    if val_raw is None or test_raw is None:
        # Create splits from the labeled train data
        split_cfg = cfg.get("split", {})
        train_ratio = float(split_cfg.get("train_ratio", 0.9))
        val_ratio = float(split_cfg.get("val_ratio", 0.05))
        test_ratio = float(split_cfg.get("test_ratio", 0.05))
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "split ratios must sum to 1.0"

        tmp = train.train_test_split(test_size=(val_ratio + test_ratio), seed=seed)
        train2 = tmp["train"]
        rest = tmp["test"]
        if test_ratio == 0:
            val2 = rest
            test2 = rest.select([])
        else:
            rest_split = rest.train_test_split(test_size=(test_ratio / (val_ratio + test_ratio)), seed=seed)
            val2 = rest_split["train"]
            test2 = rest_split["test"]
        ds = DatasetDict(train=train2, validation=val2, test=test2)

    return LoadedData(
        dataset=ds,
        label_fields=label_fields,
        text_field=text_field,
        id_field=id_field,
        tokenizer_name=cfg["model"]["hf_model_name"],
    )


def tokenize_dataset(data: LoadedData, max_length: int) -> Tuple[DatasetDict, Any]:
    tokenizer = AutoTokenizer.from_pretrained(data.tokenizer_name, use_fast=True)

    def _tok(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    tokenized = data.dataset.map(_tok, batched=True)
    # Keep 'labels' (Trainer expects it). Drop raw 'text' + optional 'id' for torch formatting.
    remove = [c for c in tokenized["train"].column_names if c in {"text", "id"}]
    tokenized = tokenized.remove_columns(remove)
    tokenized.set_format(type="torch")
    return tokenized, tokenizer
