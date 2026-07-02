#!/usr/bin/env python
"""Collect metrics.json files from cluster runs into a single CSV table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--out", default="cluster_metrics_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for path in sorted(Path(args.runs_dir).glob("*/*/metrics.json")):
        obj = json.loads(path.read_text())
        best_test = obj.get("best_test") or {}
        run_args = obj.get("args") or {}
        rows.append(
            {
                "task": obj.get("task"),
                "seed": run_args.get("seed"),
                "num_rows": obj.get("num_rows"),
                "num_valid_molecules": obj.get("num_valid_molecules"),
                "best_val_roc_auc": obj.get("best_val_roc_auc"),
                "test_roc_auc": best_test.get("roc_auc"),
                "test_avg_precision": best_test.get("avg_precision"),
                "test_accuracy@0.5": best_test.get("accuracy@0.5"),
                "metrics_path": str(path),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(df)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()

