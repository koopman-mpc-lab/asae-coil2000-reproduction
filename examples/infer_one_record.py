from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import binary_metrics
from src.models import ASAE

CASE_DIR = ROOT / "data" / "cases"
CKPT = ROOT / "checkpoints" / "asae_seed2" / "best.pt"
FEAT = ROOT / "data" / "features" / "records"
PRED = ROOT / "results" / "predictions" / "main" / "asae_seed2.csv"
YTRUE = ROOT / "results" / "predictions" / "y_true_test.csv"


def _case_json() -> Path:
    listed = sorted(CASE_DIR.glob("R*.json"))
    if not listed:
        raise FileNotFoundError(f"no case json under {CASE_DIR}")
    return listed[0]


def _load_case_x(record_id: str) -> torch.Tensor:
    path = FEAT / f"{record_id}.npz"
    blob = np.load(path)
    return torch.from_numpy(blob["x"].astype(np.float32)).unsqueeze(0)


@torch.no_grad()
def main():
    meta = json.loads(_case_json().read_text(encoding="utf-8"))
    record_id = meta["record_id"]
    print(f"record_id: {record_id}")
    print(f"split: {meta.get('split', 'test')}  seed: {meta.get('seed', 2)}")

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg", {})
    model = ASAE(
        d_z=int(cfg.get("d_z", 32)),
        dropout=float(cfg.get("dropout", 0.3)),
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    threshold = float(ckpt.get("threshold", 0.5))

    x = _load_case_x(record_id)
    out = model(x, return_aux=True)
    prob = float(torch.softmax(out["logits"], dim=-1)[0, 1])
    recon = float(torch.mean((out["xhat"] - x) ** 2))
    attn = out["attn"][0].cpu().numpy()
    y = int(np.load(FEAT / f"{record_id}.npz")["y"])

    print(f"y_true: {y}")
    print(f"p_hat: {prob:.4f}")
    print(f"threshold: {threshold:.4f}")
    print(f"y_pred: {int(prob >= threshold)}")
    print(f"recon_mse: {recon:.4f}")
    print(f"attn_min: {attn.min():.2f}")
    print(f"attn_max: {attn.max():.2f}")
    print(f"n_units_gt_0.8: {int((attn > 0.8).sum())}")
    top = np.argsort(-attn)[:8]
    print("top_units: " + ", ".join(f"u{int(i)}={attn[i]:.2f}" for i in top))

    archived = pd.read_csv(PRED)
    if "y_true" not in archived:
        y_true = pd.read_csv(YTRUE)
        archived = archived.merge(y_true, on="record_id", how="left")
    m = binary_metrics(archived["y_true"].to_numpy(), archived["y_score"].to_numpy(), threshold)
    print(
        f"seed2_test: n={m['n']}  tn={m['tn']} fp={m['fp']} fn={m['fn']} tp={m['tp']}  "
        f"acc={m['accuracy']:.4f}  prec={m['precision']:.4f}  rec={m['recall']:.4f}  "
        f"f1={m['f1']:.4f}"
    )


if __name__ == "__main__":
    main()
