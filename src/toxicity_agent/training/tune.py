from __future__ import annotations

import itertools
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    TrainingArguments,
)

from ..config import load_config
from ..logging_utils import get_logger
from ..utils import resolve_paths, set_global_seed
from ..data.dataset import load_and_prepare_dataset, tokenize_dataset
from .metrics import evaluate_multilabel
from .train import WeightedMultilabelTrainer, _compute_pos_weight

logger = get_logger(__name__)


def _now_tag() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")


def run_tune(config_path: str) -> Dict[str, Any]:
    cfg = load_config(config_path)
    seed = int(cfg.get("project", {}).get("seed", 42))
    set_global_seed(seed)

    paths_cfg = cfg.get("paths", {})
    paths = resolve_paths(
        data_dir_cfg=str(paths_cfg.get("data_dir", "")),
        artifacts_dir_cfg=str(paths_cfg.get("artifacts_dir", "")),
    )

    tune_cfg = cfg.get("tuning", {})
    if not tune_cfg.get("enabled", True):
        return {"status": "tuning disabled"}

    loaded = load_and_prepare_dataset(cfg)
    label_fields = loaded.label_fields
    tokenized, tokenizer = tokenize_dataset(loaded, max_length=int(cfg["model"]["max_length"]))

    ds_train = tokenized["train"]
    ds_val = tokenized["validation"]

    # For tuning, cap samples for speed (configurable)
    max_train = cfg["dataset"].get("max_train_samples")
    max_eval = cfg["dataset"].get("max_eval_samples")
    if max_train is not None:
        ds_train = ds_train.select(range(min(int(max_train), len(ds_train))))
    if max_eval is not None:
        ds_val = ds_val.select(range(min(int(max_eval), len(ds_val))))

    lrs = tune_cfg.get("learning_rates", [cfg["training"]["learning_rate"]])
    bss = tune_cfg.get("batch_sizes", [cfg["training"]["batch_size"]])
    eps = tune_cfg.get("num_train_epochs", [cfg["training"]["num_train_epochs"]])
    max_trials = int(tune_cfg.get("max_trials", 6))

    combos = list(itertools.product(lrs, bss, eps))[:max_trials]
    logger.info(f"Tuning trials: {len(combos)}")

    out_dir = paths.runs_dir / f"tune-{_now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    best = {"score": -1.0, "params": None}
    all_trials: List[Dict[str, Any]] = []

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = 1 / (1 + np.exp(-logits))
        res = evaluate_multilabel(labels, probs, label_fields=label_fields, threshold=0.5)
        return res.to_dict()

    # pos_weight (reuse across trials)
    pos_weight = None
    if bool(cfg["training"].get("use_pos_weight", True)):
        try:
            pos_weight = _compute_pos_weight(ds_train, n_labels=len(label_fields))
        except Exception:
            pos_weight = None

    # Precision mode
    bf16 = bool(cfg["training"].get("bf16", False))
    fp16 = bool(cfg["training"].get("fp16", False))
    if bf16 and torch.cuda.is_available():
        try:
            bf16 = torch.cuda.is_bf16_supported()
        except Exception:
            bf16 = False
    else:
        bf16 = False
    if bf16:
        fp16 = False

    for i, (lr, bs, ep) in enumerate(combos, start=1):
        trial_dir = out_dir / f"trial-{i}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        model = AutoModelForSequenceClassification.from_pretrained(
            cfg["model"]["hf_model_name"],
            num_labels=len(label_fields),
            problem_type="multi_label_classification",
        )

        if bool(cfg["training"].get("gradient_checkpointing", False)):
            try:
                model.gradient_checkpointing_enable()
            except Exception:
                pass

        args = TrainingArguments(
            output_dir=str(trial_dir / "hf"),  # local checkpoints
            per_device_train_batch_size=int(bs),
            per_device_eval_batch_size=int(cfg["training"]["eval_batch_size"]),
            learning_rate=float(lr),
            weight_decay=float(cfg["training"]["weight_decay"]),
            num_train_epochs=float(ep),
            warmup_ratio=float(cfg["training"].get("warmup_ratio", 0.0)),
            fp16=fp16,
            bf16=bf16,
            gradient_accumulation_steps=int(cfg["training"].get("gradient_accumulation_steps", 1)),
            logging_steps=int(cfg["training"].get("logging_steps", 50)),
            eval_strategy="epoch",
            save_strategy="no",
            report_to=[],
        )

        # Note: EarlyStoppingCallback is NOT used during tuning because:
        # - Tuning trials are already short (1-2 epochs each)
        # - EarlyStoppingCallback requires load_best_model_at_end + save_strategy != "no"
        #   which conflicts with our lightweight tuning setup.
        trainer = WeightedMultilabelTrainer(
            pos_weight=pos_weight,
            model=model,
            args=args,
            train_dataset=ds_train,
            eval_dataset=ds_val,
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        eval_metrics = trainer.evaluate()

        score = float(eval_metrics.get("eval_f1_micro", -1.0))
        trial = {
            "trial": i,
            "learning_rate": lr,
            "batch_size": bs,
            "num_train_epochs": ep,
            "eval_f1_micro": score,
            "eval_metrics": eval_metrics,
        }
        all_trials.append(trial)

        if score > best["score"]:
            best = {"score": score, "params": {"learning_rate": lr, "batch_size": bs, "num_train_epochs": ep}}

        logger.info(f"Trial {i}: f1_micro={score:.4f} best={best['params']}")  # type: ignore

    results = {"best": best, "trials": all_trials, "label_fields": label_fields}
    (out_dir / "tune_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info(f"Tune results saved to {out_dir / 'tune_results.json'}")
    return results
