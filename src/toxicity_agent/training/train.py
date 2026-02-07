from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from ..config import load_config
from ..data.dataset import load_and_prepare_dataset, tokenize_dataset
from ..logging_utils import get_logger
from ..utils import resolve_paths, set_global_seed
from ..versioning.model_metadata import write_model_metadata
from .metrics import evaluate_multilabel

logger = get_logger(__name__)


def _make_run_dir(runs_dir: Path) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    run_dir = runs_dir / f"run-{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


class WeightedMultilabelTrainer(Trainer):
    """Trainer using BCEWithLogitsLoss + (optional) pos_weight for class imbalance.

    Toxicity datasets typically have rare labels (e.g., threat). pos_weight helps the model
    pay attention to rare positives.

    Note: This is still a simple baseline technique; for production you'd likely combine
    it with threshold tuning, calibration, and careful evaluation.
    """

    def __init__(self, pos_weight: np.ndarray | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pos_weight = None
        if pos_weight is not None:
            self._pos_weight = torch.tensor(pos_weight, dtype=torch.float32)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits

        if labels is None:
            loss = outputs.loss
        else:
            labels_f = labels.float()
            if self._pos_weight is not None:
                pos_w = self._pos_weight.to(logits.device)
                loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
            else:
                loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels_f)

        return (loss, outputs) if return_outputs else loss


def _compute_pos_weight(ds_train, n_labels: int) -> np.ndarray:
    """Compute pos_weight = (neg+1)/(pos+1) per label."""
    labels = np.array([x.numpy() for x in ds_train["labels"]], dtype=np.float32)
    pos = labels.sum(axis=0)
    neg = labels.shape[0] - pos
    pos_weight = (neg + 1.0) / (pos + 1.0)
    if pos_weight.shape[0] != n_labels:
        pos_weight = np.ones((n_labels,), dtype=np.float32)
    return pos_weight.astype(np.float32)


def run_train(config_path: str) -> Tuple[Path, Dict[str, Any]]:
    """Train a multi-label transformer and save artifacts to models/finetuned/.

    Also writes model_metadata.json (versioning) next to the model.
    """
    cfg = load_config(config_path)
    seed = int(cfg.get("project", {}).get("seed", 42))
    set_global_seed(seed)

    paths_cfg = cfg.get("paths", {})
    paths = resolve_paths(
        data_dir_cfg=str(paths_cfg.get("data_dir", "")),
        artifacts_dir_cfg=str(paths_cfg.get("artifacts_dir", "")),
    )

    run_dir = _make_run_dir(paths.runs_dir)
    # Save a copy of config used for this run
    (run_dir / "config.yaml").write_text(Path(config_path).read_text(encoding="utf-8"), encoding="utf-8")

    loaded = load_and_prepare_dataset(cfg)
    tokenized, tokenizer = tokenize_dataset(loaded, max_length=int(cfg["model"]["max_length"]))

    max_train = cfg["dataset"].get("max_train_samples")
    max_eval = cfg["dataset"].get("max_eval_samples")

    ds_train = tokenized["train"]
    ds_val = tokenized["validation"]
    ds_test = tokenized["test"]

    if max_train is not None:
        ds_train = ds_train.select(range(min(int(max_train), len(ds_train))))
    if max_eval is not None:
        ds_val = ds_val.select(range(min(int(max_eval), len(ds_val))))
        ds_test = ds_test.select(range(min(int(max_eval), len(ds_test))))

    label_fields = loaded.label_fields
    num_labels = len(label_fields)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model"]["hf_model_name"],
        num_labels=num_labels,
        problem_type="multi_label_classification",
    )

    train_args = cfg["training"]

    # Gradient checkpointing (helps with large models / long sequences)
    if bool(train_args.get("gradient_checkpointing", False)):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            logger.warning("Could not enable gradient checkpointing for this model.")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Mixed precision: prefer bf16 on H100, fallback to fp16.
    bf16 = bool(train_args.get("bf16", False))
    fp16 = bool(train_args.get("fp16", False))
    if bf16 and torch.cuda.is_available():
        try:
            bf16 = torch.cuda.is_bf16_supported()
        except Exception:
            bf16 = False
    else:
        bf16 = False
    if bf16:
        fp16 = False

    args = TrainingArguments(
        output_dir=str(run_dir / "hf_checkpoints"),
        per_device_train_batch_size=int(train_args["batch_size"]),
        per_device_eval_batch_size=int(train_args["eval_batch_size"]),
        learning_rate=float(train_args["learning_rate"]),
        weight_decay=float(train_args["weight_decay"]),
        num_train_epochs=float(train_args["num_train_epochs"]),
        warmup_ratio=float(train_args.get("warmup_ratio", 0.0)),
        fp16=fp16,
        bf16=bf16,
        gradient_accumulation_steps=int(train_args.get("gradient_accumulation_steps", 1)),
        logging_steps=int(train_args.get("logging_steps", 50)),
        evaluation_strategy=str(train_args.get("eval_strategy", "steps")),
        eval_steps=int(train_args.get("eval_steps", 200)),
        save_steps=int(train_args.get("save_steps", 200)),
        save_total_limit=int(train_args.get("save_total_limit", 2)),
        load_best_model_at_end=True,
        metric_for_best_model=str(train_args.get("metric_for_best_model", "f1_micro")),
        greater_is_better=bool(train_args.get("greater_is_better", True)),
        report_to=[],
        save_safetensors=True,
        seed=seed,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = 1 / (1 + np.exp(-logits))
        res = evaluate_multilabel(labels, probs, label_fields=label_fields, threshold=0.5)
        return res.to_dict()

    # pos_weight for imbalance
    pos_weight = None
    if bool(train_args.get("use_pos_weight", True)):
        try:
            pos_weight = _compute_pos_weight(ds_train, n_labels=num_labels)
        except Exception:
            pos_weight = None

    callbacks = None
    if bool(train_args.get("early_stopping", True)):
        callbacks = [EarlyStoppingCallback(early_stopping_patience=int(train_args.get("early_stopping_patience", 3)))]

    trainer = WeightedMultilabelTrainer(
        pos_weight=pos_weight,
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    logger.info(f"Training started. run_dir={run_dir}")
    trainer.train()

    # Evaluate on test
    test_metrics: Dict[str, Any] = {}
    if len(ds_test) > 0:
        test_metrics = trainer.evaluate(eval_dataset=ds_test, metric_key_prefix="test")

    # Save final model
    out_model_dir = paths.models_dir / "finetuned" / run_dir.name
    out_model_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(out_model_dir))
    tokenizer.save_pretrained(str(out_model_dir))

    # Save label fields
    (out_model_dir / "label_fields.json").write_text(json.dumps(label_fields, indent=2), encoding="utf-8")

    # Write model metadata (versioning)
    best_metric_name = str(args.metric_for_best_model) if args.metric_for_best_model else None
    best_metric_value = None
    try:
        if trainer.state.best_metric is not None:
            best_metric_value = float(trainer.state.best_metric)
    except Exception:
        best_metric_value = None

    write_model_metadata(
        model_dir=out_model_dir,
        repo_root=paths.repo_root,
        cfg=cfg,
        run_id=run_dir.name,
        test_metrics=test_metrics,
        best_metric_name=best_metric_name,
        best_metric_value=best_metric_value,
        n_train=len(ds_train),
        n_val=len(ds_val),
        n_test=len(ds_test),
        model_type="finetuned_transformer",
    )

    # Update latest pointer (copy for portability; symlink may break on some OS)
    latest_dir = paths.models_dir / "finetuned" / "latest"
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(out_model_dir, latest_dir)

    # Save a run manifest
    metrics_path = run_dir / "metrics_test.json"
    metrics_path.write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")

    result = {
        "run_dir": str(run_dir),
        "model_dir": str(out_model_dir),
        "latest_dir": str(latest_dir),
        "test_metrics": test_metrics,
        "label_fields": label_fields,
        "pos_weight": pos_weight.tolist() if pos_weight is not None else None,
        "precision_mode": "bf16" if bf16 else ("fp16" if fp16 else "fp32"),
        "best_metric_name": best_metric_name,
        "best_metric_value": best_metric_value,
    }
    (run_dir / "run_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    return out_model_dir, result
