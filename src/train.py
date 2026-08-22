from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data.dataset import CoilDataset, collate_records
from .losses import total_loss
from .metrics import best_f1_threshold, binary_metrics
from .models import ASAE
from .utils import ROOT, load_yaml, param_line, set_seed


@torch.no_grad()
def _scores(model, loader, device):
    model.eval()
    ys, ps, ids = [], [], []
    for batch in loader:
        x = batch["x"].to(device)
        out = model(x, return_aux=True)
        prob = torch.softmax(out["logits"], dim=-1)[:, 1].cpu().numpy()
        ys.append(batch["y"].numpy())
        ps.append(prob)
        ids.extend(batch["record_id"])
    return np.concatenate(ys), np.concatenate(ps), ids


def train_one(cfg: dict, smoke: bool = False) -> dict:
    set_seed(int(cfg.get("seed", 0)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ASAE(
        d_z=int(cfg.get("d_z", 32)),
        dropout=float(cfg.get("dropout", 0.3)),
        use_attention=bool(cfg.get("use_attention", True)),
        use_decoder=bool(cfg.get("use_decoder", True)),
        use_residual=bool(cfg.get("use_residual", True)),
        use_dropout=bool(cfg.get("use_dropout", True)),
    ).to(device)
    print(param_line(model), flush=True)

    train_ds = CoilDataset("train")
    val_ds = CoilDataset("val")
    bs = 32 if smoke else int(cfg.get("batch_size", 256))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate_records)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=collate_records)

    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg.get("lr", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    epochs = 2 if smoke else int(cfg.get("max_epochs", 200))
    patience = int(cfg.get("patience", 20))
    y_tr = train_ds.y
    n0 = max(1, int((y_tr == 0).sum()))
    n1 = max(1, int((y_tr == 1).sum()))
    w0, w1 = 0.5 * (n0 + n1) / n0, 0.5 * (n0 + n1) / n1

    best_auc, best_state, bad, best_epoch = -1.0, None, 0, 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            out = model(x, return_aux=True)
            loss, _ = total_loss(
                out["logits"],
                y,
                x,
                out["xhat"],
                out["z"],
                w0,
                w1,
                alpha=float(cfg.get("alpha", 0.5)),
                beta=float(cfg.get("beta", 1e-2)),
                use_ent=bool(cfg.get("use_ent", True)),
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        yv, pv, _ = _scores(model, val_loader, device)
        from sklearn.metrics import roc_auc_score

        vauc = float(roc_auc_score(yv, pv))
        print(
            f"epoch {epoch:03d}  train_loss={np.mean(losses):.4f}  val_auc={vauc:.4f}",
            flush=True,
        )
        if vauc > best_auc:
            best_auc, best_epoch, bad = vauc, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"early stop at epoch {epoch}, best {best_epoch}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    thr = best_f1_threshold(yv, pv)
    test_loader = DataLoader(CoilDataset("test"), batch_size=bs, shuffle=False, collate_fn=collate_records)
    yt, pt, ids = _scores(model, test_loader, device)
    metrics = binary_metrics(yt, pt, thr)
    metrics.update({"best_epoch": best_epoch, "best_val_auc": best_auc, "seconds": time.time() - t0})
    out_dir = Path(cfg.get("ckpt_dir", ROOT / "checkpoints" / f"asae_seed{cfg.get('seed', 0)}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": best_epoch,
            "val_auc": best_auc,
            "threshold": thr,
            "cfg": cfg,
        },
        out_dir / "best.pt",
    )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "asae.yaml"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    metrics = train_one(cfg, smoke=args.smoke)
    print(metrics)


if __name__ == "__main__":
    main()
