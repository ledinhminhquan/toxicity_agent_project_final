"""CLI entry point for toxicity-agent.

All heavy imports (torch, transformers, etc.) are done lazily inside command
handlers so that lightweight commands (--help) load instantly and tests that
don't need ML deps still work.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def _ts() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")


# ---------------------------------------------------------------------------
# Command handlers (lazy imports inside each function)
# ---------------------------------------------------------------------------

def _serve(config_path: str, host: str, port: int) -> None:
    import uvicorn
    os.environ["TOXICITY_INFER_CONFIG"] = config_path
    uvicorn.run("toxicity_agent.api.main:app", host=host, port=port, reload=False)


def _demo_agent(config_path: str) -> None:
    from .agent.factory import build_agent

    agent = build_agent(config_path)
    print("\n--- Moderation Agent Demo ---")
    print("Type a message and press Enter. Type 'exit' to quit.\n")
    while True:
        text = input("> ").strip()
        if text.lower() in {"exit", "quit"}:
            break
        decision = agent.moderate(text)
        print(json.dumps(decision.to_dict(), indent=2, ensure_ascii=False))
        print("")


def _report(config_path: str) -> None:
    from .config import load_config
    from .monitoring.daily_report import generate_report
    from .utils import resolve_paths

    cfg = load_config(config_path)
    paths = resolve_paths(
        data_dir_cfg=str(cfg.get("paths", {}).get("data_dir", "")),
        artifacts_dir_cfg=str(cfg.get("paths", {}).get("artifacts_dir", "")),
    )
    pred_log_name = str(cfg.get("logging", {}).get("predictions_log_name", "predictions.jsonl"))
    pred_log = paths.runs_dir / pred_log_name
    report = generate_report(pred_log)
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))


def _benchmark(config_path: str, n: int, warmup: int, out: str | None) -> None:
    from .agent.factory import build_agent
    from .analysis.latency import benchmark_callable, save_latency_report
    from .config import load_config
    from .utils import resolve_paths

    cfg = load_config(config_path)
    paths = resolve_paths(
        data_dir_cfg=str(cfg.get("paths", {}).get("data_dir", "")),
        artifacts_dir_cfg=str(cfg.get("paths", {}).get("artifacts_dir", "")),
    )
    agent = build_agent(config_path)
    texts = ["This is a safe message. Please be respectful."] * int(n)
    stats = benchmark_callable(agent.moderate, texts=texts, warmup=int(warmup))
    report = {"mode": "agent_end_to_end", "stats": stats.to_dict(), "n": int(n), "warmup": int(warmup)}
    out_path = Path(out) if out else (paths.runs_dir / "benchmarks" / f"benchmark-{_ts()}.json")
    save_latency_report(out_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def _run_error_analysis_cmd(
    train_config_path: str, split: str, threshold: float, max_samples: int | None, model_kind: str, out: str | None
) -> None:
    from .analysis.error_analysis import run_error_analysis, save_error_analysis
    from .config import load_config
    from .utils import resolve_paths

    cfg = load_config(train_config_path)
    paths = resolve_paths(
        data_dir_cfg=str(cfg.get("paths", {}).get("data_dir", "")),
        artifacts_dir_cfg=str(cfg.get("paths", {}).get("artifacts_dir", "")),
    )
    report = run_error_analysis(
        train_config_path=train_config_path,
        split=split,
        threshold=float(threshold),
        max_samples=max_samples,
        model_kind=model_kind,
    )
    out_path = Path(out) if out else (paths.runs_dir / "error_analysis" / f"error-analysis-{_ts()}.json")
    save_error_analysis(out_path, report)
    print(json.dumps({"saved_to": str(out_path)}, indent=2, ensure_ascii=False))


def _run_fairness_cmd(
    train_config_path: str,
    fairness_cfg: str,
    split: str,
    threshold: float,
    max_samples: int | None,
    model_kind: str,
    out: str | None,
) -> None:
    from .analysis.fairness import evaluate_fairness_slices, save_fairness_report
    from .config import load_config
    from .utils import resolve_paths

    cfg = load_config(train_config_path)
    paths = resolve_paths(
        data_dir_cfg=str(cfg.get("paths", {}).get("data_dir", "")),
        artifacts_dir_cfg=str(cfg.get("paths", {}).get("artifacts_dir", "")),
    )
    report = evaluate_fairness_slices(
        train_config_path=train_config_path,
        fairness_slices_path=Path(fairness_cfg),
        model_kind=model_kind,
        split=split,
        threshold=float(threshold),
        max_samples=max_samples,
    )
    out_path = Path(out) if out else (paths.runs_dir / "fairness" / f"fairness-{_ts()}.json")
    save_fairness_report(out_path, report)
    print(json.dumps({"saved_to": str(out_path)}, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Main CLI definition
# ---------------------------------------------------------------------------

def main() -> None:
    from .logging_utils import setup_logging
    setup_logging()

    parser = argparse.ArgumentParser(prog="toxicity-agent")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # --- train ---
    p_train = sub.add_parser("train", help="Train a fine-tuned transformer model")
    p_train.add_argument("--config", required=True)

    # --- tune ---
    p_tune = sub.add_parser("tune", help="Basic hyperparameter tuning")
    p_tune.add_argument("--config", required=True)

    # --- eval ---
    p_eval = sub.add_parser("eval", help="Evaluate baselines and fine-tuned model")
    p_eval.add_argument("--config", required=True)

    # --- serve ---
    p_serve = sub.add_parser("serve", help="Run FastAPI server")
    p_serve.add_argument("--config", required=True)
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)

    # --- demo-agent ---
    p_demo = sub.add_parser("demo-agent", help="Interactive agent demo")
    p_demo.add_argument("--config", required=True)

    # --- report ---
    p_report = sub.add_parser("report", help="Generate a simple monitoring report from logs")
    p_report.add_argument("--config", required=True)

    # --- benchmark ---
    p_bench = sub.add_parser("benchmark", help="Measure latency (p50/p95/p99) for the end-to-end agent")
    p_bench.add_argument("--config", required=True)
    p_bench.add_argument("--n", type=int, default=300, help="Number of synthetic requests")
    p_bench.add_argument("--warmup", type=int, default=10)
    p_bench.add_argument("--out", default=None, help="Output JSON path (default: runs/benchmark-*.json)")

    # --- error-analysis ---
    p_err = sub.add_parser(
        "error-analysis",
        help="Run privacy-preserving error analysis (no raw text written; only aggregates + hashes).",
    )
    p_err.add_argument("--config", required=True, help="Training config (e.g., configs/train_final.yaml)")
    p_err.add_argument("--split", default="test", choices=["train", "validation", "test"])
    p_err.add_argument("--threshold", type=float, default=0.5)
    p_err.add_argument("--max-samples", type=int, default=None)
    p_err.add_argument("--model-kind", default="finetuned", choices=["finetuned", "detoxify-unbiased"])
    p_err.add_argument("--out", default=None, help="Output JSON path (default: runs/error-analysis-*.json)")

    # --- fairness ---
    p_fair = sub.add_parser("fairness", help="Evaluate fairness slices based on identity-term mentions.")
    p_fair.add_argument("--config", required=True, help="Training config (e.g., configs/train_final.yaml)")
    p_fair.add_argument("--fairness-config", default="configs/fairness_slices.yaml")
    p_fair.add_argument("--split", default="test", choices=["train", "validation", "test"])
    p_fair.add_argument("--threshold", type=float, default=0.5)
    p_fair.add_argument("--max-samples", type=int, default=None)
    p_fair.add_argument("--model-kind", default="finetuned", choices=["finetuned", "detoxify-unbiased", "detoxify-multilingual"])
    p_fair.add_argument("--out", default=None, help="Output JSON path (default: runs/fairness-*.json)")

    # --- threshold-search ---
    p_th = sub.add_parser("threshold-search", help="Search thresholds on validation split and save JSON next to model")
    p_th.add_argument("--config", required=True, help="Training config (e.g., configs/train_final.yaml)")
    p_th.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    p_th.add_argument("--out", default=None, help="Output JSON path (default: models/finetuned/latest/thresholds_val.json)")

    # --- generate-report ---
    p_gr = sub.add_parser("generate-report", help="Generate PDF report skeleton (10-15 pages) from latest artifacts")
    p_gr.add_argument("--train-config", required=True)
    p_gr.add_argument("--infer-config", required=True)
    p_gr.add_argument("--out", default=None, help="Output PDF path")
    p_gr.add_argument("--title", default="Hate Speech & Toxicity Detection System")
    p_gr.add_argument("--author", default="(fill) Student / Team")

    # --- generate-slides ---
    p_gs = sub.add_parser("generate-slides", help="Generate PPTX slide deck (10-15 slides) from latest artifacts")
    p_gs.add_argument("--train-config", required=True)
    p_gs.add_argument("--infer-config", required=True)
    p_gs.add_argument("--out", default=None, help="Output PPTX path")
    p_gs.add_argument("--title", default="Hate Speech & Toxicity Detection System")
    p_gs.add_argument("--author", default="(fill) Student / Team")

    # --- autopilot ---
    p_auto = sub.add_parser("autopilot", help="One-button pipeline: train+eval+analysis+report+slides")
    p_auto.add_argument("--train-config", required=True)
    p_auto.add_argument("--infer-config", required=True)
    p_auto.add_argument("--fairness-config", default="configs/fairness_slices.yaml")
    p_auto.add_argument("--out-dir", default=None, help="Submission output directory")
    p_auto.add_argument("--do-tune", action="store_true", help="Run tuning before training")
    p_auto.add_argument("--tune-config", default=None, help="Tuning config (default: use train-config)")
    p_auto.add_argument("--skip-train", action="store_true", help="Skip training step")
    p_auto.add_argument("--skip-eval", action="store_true", help="Skip evaluation step")
    p_auto.add_argument("--skip-thresholds", action="store_true", help="Skip threshold search step")
    p_auto.add_argument("--skip-benchmark", action="store_true", help="Skip latency benchmark step")
    p_auto.add_argument("--skip-error-analysis", action="store_true", help="Skip error analysis step")
    p_auto.add_argument("--skip-fairness", action="store_true", help="Skip fairness evaluation step")
    p_auto.add_argument("--title", default="Hate Speech & Toxicity Detection System")
    p_auto.add_argument("--author", default="(fill) Student / Team")

    args = parser.parse_args()

    # ---- Dispatch ----
    if args.cmd == "train":
        from .training.train import run_train
        _, result = run_train(args.config)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.cmd == "tune":
        from .training.tune import run_tune
        res = run_tune(args.config)
        print(json.dumps(res, indent=2, ensure_ascii=False))

    elif args.cmd == "eval":
        from .training.evaluate import run_eval
        res = run_eval(args.config)
        print(json.dumps(res, indent=2, ensure_ascii=False))

    elif args.cmd == "serve":
        _serve(args.config, host=args.host, port=args.port)

    elif args.cmd == "demo-agent":
        _demo_agent(args.config)

    elif args.cmd == "report":
        _report(args.config)

    elif args.cmd == "benchmark":
        _benchmark(args.config, n=args.n, warmup=args.warmup, out=args.out)

    elif args.cmd == "error-analysis":
        _run_error_analysis_cmd(
            train_config_path=args.config,
            split=args.split,
            threshold=args.threshold,
            max_samples=args.max_samples,
            model_kind=args.model_kind,
            out=args.out,
        )

    elif args.cmd == "fairness":
        _run_fairness_cmd(
            train_config_path=args.config,
            fairness_cfg=args.fairness_config,
            split=args.split,
            threshold=args.threshold,
            max_samples=args.max_samples,
            model_kind=args.model_kind,
            out=args.out,
        )

    elif args.cmd == "threshold-search":
        from .analysis.thresholds import threshold_search
        res = threshold_search(train_config_path=args.config, split=args.split, out_path=args.out)
        print(json.dumps(res, indent=2, ensure_ascii=False))

    elif args.cmd == "generate-report":
        from .autoreport.artifact_loader import collect_artifacts
        from .autoreport.report_pdf import generate_report_pdf
        from .config import load_config
        from .utils import resolve_paths

        cfg = load_config(args.train_config)
        paths = resolve_paths(
            data_dir_cfg=str(cfg.get("paths", {}).get("data_dir", "")),
            artifacts_dir_cfg=str(cfg.get("paths", {}).get("artifacts_dir", "")),
        )
        bundle = collect_artifacts(
            runs_dir=paths.runs_dir,
            models_dir=paths.models_dir,
            train_config_path=Path(args.train_config),
            infer_config_path=Path(args.infer_config),
        )
        out_path = Path(args.out) if args.out else (paths.artifacts_dir / "submission" / f"submission-{_ts()}" / "report.pdf")
        generate_report_pdf(bundle=bundle, out_path=out_path, title=args.title, author=args.author)
        print(json.dumps({"saved_to": str(out_path)}, indent=2, ensure_ascii=False))

    elif args.cmd == "generate-slides":
        from .autoreport.artifact_loader import collect_artifacts
        from .autoreport.slides_pptx import generate_slides_pptx
        from .config import load_config
        from .utils import resolve_paths

        cfg = load_config(args.train_config)
        paths = resolve_paths(
            data_dir_cfg=str(cfg.get("paths", {}).get("data_dir", "")),
            artifacts_dir_cfg=str(cfg.get("paths", {}).get("artifacts_dir", "")),
        )
        bundle = collect_artifacts(
            runs_dir=paths.runs_dir,
            models_dir=paths.models_dir,
            train_config_path=Path(args.train_config),
            infer_config_path=Path(args.infer_config),
        )
        out_path = Path(args.out) if args.out else (paths.artifacts_dir / "submission" / f"submission-{_ts()}" / "slides.pptx")
        generate_slides_pptx(bundle=bundle, out_path=out_path, title=args.title, author_line=args.author)
        print(json.dumps({"saved_to": str(out_path)}, indent=2, ensure_ascii=False))

    elif args.cmd == "autopilot":
        from .automation.autopilot import run_autopilot
        res = run_autopilot(
            train_config_path=args.train_config,
            infer_config_path=args.infer_config,
            fairness_config_path=args.fairness_config,
            out_dir=args.out_dir,
            do_tune=bool(args.do_tune),
            tune_config_path=args.tune_config,
            do_train=not bool(args.skip_train),
            do_eval=not bool(args.skip_eval),
            do_threshold_search=not bool(args.skip_thresholds),
            do_benchmark=not bool(args.skip_benchmark),
            do_error_analysis=not bool(args.skip_error_analysis),
            do_fairness=not bool(args.skip_fairness),
            report_title=args.title,
            report_author=args.author,
        )
        print(json.dumps(res.to_dict(), indent=2, ensure_ascii=False))

    else:
        raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
