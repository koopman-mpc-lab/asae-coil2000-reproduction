from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import binary_metrics
from .utils import ROOT


def load_y_true(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path or ROOT / "results" / "predictions" / "y_true_test.csv")
    return pd.read_csv(path)


def load_pred(path: str | Path) -> pd.DataFrame:
    pred = pd.read_csv(path)
    if "y_true" not in pred.columns:
        y_true = load_y_true()
        pred = pred.merge(y_true, on="record_id", how="left")
    return pred


def metrics_from_pred(path: str | Path, threshold: float | None = None) -> dict:
    df = load_pred(path)
    if threshold is None:
        if "threshold" in df.columns:
            threshold = float(df["threshold"].iloc[0])
        elif "y_pred" in df.columns:
            # recover a threshold consistent with archived hard labels
            pos = df.loc[df["y_pred"] == 1, "y_score"]
            neg = df.loc[df["y_pred"] == 0, "y_score"]
            if len(pos) and len(neg):
                threshold = float(0.5 * (pos.min() + neg.max()))
            else:
                threshold = 0.5
        else:
            threshold = 0.5
    return binary_metrics(df["y_true"].to_numpy(), df["y_score"].to_numpy(), float(threshold))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    m = metrics_from_pred(args.pred, args.threshold)
    for k, v in m.items():
        if isinstance(v, float):
            print(f"{k:12s} {v:.4f}")
        else:
            print(f"{k:12s} {v}")


if __name__ == "__main__":
    main()
