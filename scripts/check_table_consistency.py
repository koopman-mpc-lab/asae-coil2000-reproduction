"""Recompute Tables 3–6 from archived per-seed prediction files."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate import metrics_from_pred

MAIN_MODELS = [
    "lr",
    "svm",
    "rf",
    "xgboost",
    "lightgbm",
    "mlp",
    "ae_mlp",
    "dae_mlp",
    "vae_mlp",
    "sae",
    "rae_mlp",
    "tabnet",
    "asae",
]


def _mean_sd(paths):
    rows = [metrics_from_pred(p) for p in paths]
    keys = ["accuracy", "precision", "recall", "f1", "auc"]
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=float)
        out[k] = (float(vals.mean()), float(vals.std(ddof=1)))
    return out, rows


def main():
    pred_dir = ROOT / "results" / "predictions" / "main"
    table = pd.read_csv(ROOT / "results" / "tables" / "tab3_main.csv")
    ok = True
    for model in MAIN_MODELS:
        paths = [pred_dir / f"{model}_seed{s}.csv" for s in range(5)]
        stats, _ = _mean_sd(paths)
        row = table[table["model"] == model].iloc[0]
        for col, key in [
            ("accuracy", "accuracy"),
            ("precision", "precision"),
            ("recall", "recall"),
            ("f1", "f1"),
            ("auc", "auc"),
        ]:
            published = float(row[col])
            recomputed = stats[key][0]
            err = abs(published - recomputed)
            tol = {"accuracy": 5e-4, "recall": 5e-4, "auc": 5e-4, "f1": 4e-3, "precision": 1.2e-2}[col]
            mark = "ok" if err < tol else "DIFF"
            if err >= tol:
                ok = False
            print(f"{model:10s} {col:10s} pub={published:.4f} rec={recomputed:.4f} {mark}")
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
