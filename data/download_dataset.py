#!/usr/bin/env python
"""Download and cache the dataset locally (optional helper).

This script is optional because HuggingFace `datasets` will cache automatically.
However, in Google Colab you may want to cache dataset files into Google Drive
to avoid re-downloading across sessions.

Example:
  python data/download_dataset.py --dataset thesofakillers/jigsaw-toxic-comment-classification-challenge \
      --out_dir /content/drive/MyDrive/NLP_Project/toxicity_agent/artifacts/data/raw
"""

from __future__ import annotations

import argparse
from pathlib import Path
from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="HuggingFace dataset name")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save parquet files")
    parser.add_argument("--max_rows", type=int, default=None, help="Optional row limit per split")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset)

    for split_name, split in ds.items():
        if args.max_rows is not None:
            split = split.select(range(min(args.max_rows, len(split))))
        split_path = out_dir / f"{split_name}.parquet"
        split.to_parquet(str(split_path))
        print(f"Saved {split_name} -> {split_path} ({len(split)} rows)")


if __name__ == "__main__":
    main()
