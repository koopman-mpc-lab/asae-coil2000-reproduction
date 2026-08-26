from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .data.coil_loader import load_coil_table
from .losses import total_loss
from .metrics import best_f1_threshold, binary_metrics
from .models import ASAE
from .utils import ROOT, load_yaml, param_line, set_seed

LOG = logging.getLogger("asae.train")


class ArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, ids: np.ndarray):
        self.x = np.asarray(x, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        self.ids = np.asarray(ids).astype(str)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> dict[str, Any]:
        return {
            "x": torch.from_numpy(self.x[i]),
            "y": torch.tensor(int(self.y[i]), dtype=torch.long),
            "record_id": str(self.ids[i]),
        }


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "x": torch.stack([row["x"] for row in batch]),
        "y": torch.stack([row["y"] for row in batch]),
        "record_id": [row["record_id"] for row in batch],
    }


@dataclass
class PreparedData:
    train: ArrayDataset
    val: ArrayDataset
    test: ArrayDataset
    mixed_layout: tuple[int, list[int]] | None
    original_train_y: np.ndarray
    split_path: Path
    preprocessing: dict[str, Any]


def _resolve_path(value: str | Path, default_parent: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else default_parent / path


def _sampling_requested(cfg: dict) -> bool:
    return bool(cfg.get("smote", cfg.get("smote_ratio") is not None)) and float(
        cfg.get("smote_ratio", 0.0)
    ) > 0


def _resample_numeric(
    x: np.ndarray, y: np.ndarray, ratio: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    current = float((y == 1).sum()) / max(1, int((y == 0).sum()))
    if ratio <= current:
        LOG.warning("SMOTE ratio %.4f is not above current ratio %.4f; skipping", ratio, current)
        return x, y
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError as exc:
        raise RuntimeError("SMOTE requested; install imbalanced-learn") from exc
    sampler = SMOTE(sampling_strategy=ratio, random_state=seed, k_neighbors=5)
    xr, yr = sampler.fit_resample(x, y)
    return np.asarray(xr, dtype=np.float32), np.asarray(yr, dtype=np.int64)


def prepare_data(cfg: dict) -> PreparedData:
    """Fit preprocessing and optional oversampling on the training rows only."""
    x_raw, y, ids = load_coil_table()
    split_value = cfg.get("split_path", "primary.npz")
    split_path = _resolve_path(split_value, ROOT / "data" / "splits")
    if not split_path.exists():
        raise FileNotFoundError(
            f"missing split file {split_path}; run scripts/generate_splits.py first"
        )
    split = np.load(split_path, allow_pickle=True)
    indices = {
        name: np.asarray(split[f"{name}_idx"], dtype=int)
        for name in ("train", "val", "test")
    }
    if set(indices["train"]) & set(indices["val"]) or set(indices["train"]) & set(indices["test"]):
        raise ValueError("split partitions overlap")

    feature_name = str(cfg.get("features", "x_std.npz"))
    mixed = bool(cfg.get("mixed_recon", False)) or "mixed" in feature_name.lower()
    seed = int(cfg.get("seed", 0))
    ratio = float(cfg.get("smote_ratio", 1.0 / 3.0))
    original_train_y = y[indices["train"]].copy()

    if mixed:
        from .data.mixed import NOMINAL_IDX, fit_mixed_spec, smotenc_expand, transform_mixed

        configured_cat = tuple(int(v) for v in cfg.get("smotenc_categorical", NOMINAL_IDX))
        if configured_cat != tuple(NOMINAL_IDX):
            raise ValueError(
                f"smotenc_categorical must match the nominal columns {list(NOMINAL_IDX)}"
            )
        spec = fit_mixed_spec(x_raw, indices["train"])
        x_all = transform_mixed(x_raw, spec)
        numeric_dim = len(spec["numeric_cols"])
        block_sizes = [len(spec["categories"][j]) for j in NOMINAL_IDX]
        x_train, y_train = x_raw[indices["train"]], original_train_y
        if _sampling_requested(cfg):
            x_train, y_train = smotenc_expand(x_train, y_train, ratio=ratio, seed=seed)
        x_train = transform_mixed(x_train, spec)
        preprocessing = {
            "kind": "mixed",
            "numeric_dim": numeric_dim,
            "block_sizes": block_sizes,
            "smotenc_categorical": list(configured_cat),
        }
        mixed_layout: tuple[int, list[int]] | None = (numeric_dim, block_sizes)
    else:
        tr_raw = x_raw[indices["train"]]
        mean = tr_raw.mean(axis=0)
        scale = tr_raw.std(axis=0)
        scale = np.where(scale < 1e-6, 1.0, scale)
        x_all = ((x_raw - mean) / scale).astype(np.float32)
        x_train, y_train = x_all[indices["train"]], original_train_y
        if _sampling_requested(cfg):
            x_train, y_train = _resample_numeric(x_train, y_train, ratio, seed)
        preprocessing = {
            "kind": "standard",
            "mean": mean.astype(float).tolist(),
            "scale": scale.astype(float).tolist(),
        }
        mixed_layout = None

    train_ids = np.asarray([f"SMOTE_{i:06d}" for i in range(len(y_train))])
    n_original = len(indices["train"])
    train_ids[:n_original] = ids[indices["train"]]
    return PreparedData(
        train=ArrayDataset(x_train, y_train, train_ids),
        val=ArrayDataset(x_all[indices["val"]], y[indices["val"]], ids[indices["val"]]),
        test=ArrayDataset(x_all[indices["test"]], y[indices["test"]], ids[indices["test"]]),
        mixed_layout=mixed_layout,
        original_train_y=original_train_y,
        split_path=split_path,
        preprocessing=preprocessing,
    )


def build_model(cfg: dict, d_in: int | None = None) -> ASAE:
    return ASAE(
        d_in=int(d_in if d_in is not None else cfg.get("d_in", 85)),
        d_h1=int(cfg.get("d_h1", 128)),
        d_h2=int(cfg.get("d_h2", 64)),
        d_z=int(cfg.get("d_z", 32)),
        d_cls=int(cfg.get("d_cls", 16)),
        reduction=int(cfg.get("reduction", 8)),
        dropout=float(cfg.get("dropout", 0.3)),
        use_attention=bool(cfg.get("use_attention", True)),
        use_decoder=bool(cfg.get("use_decoder", True)),
        use_residual=bool(cfg.get("use_residual", True)),
        use_dropout=bool(cfg.get("use_dropout", True)),
    )


def _class_weights(setting: Any, y: np.ndarray) -> tuple[float, float]:
    if setting in (False, None):
        return 1.0, 1.0
    if isinstance(setting, dict):
        return float(setting.get(0, setting.get("0", 1.0))), float(
            setting.get(1, setting.get("1", 1.0))
        )
    if isinstance(setting, (list, tuple)):
        if len(setting) != 2:
            raise ValueError("class_weight list must contain [w0, w1]")
        return float(setting[0]), float(setting[1])
    n0, n1 = max(1, int((y == 0).sum())), max(1, int((y == 1).sum()))
    n = n0 + n1
    return 0.5 * n / n0, 0.5 * n / n1


def _loss_kwargs(cfg: dict, mixed_layout) -> dict[str, Any]:
    no_recon = bool(cfg.get("no_recon", False))
    return {
        "w0": 1.0,
        "w1": 1.0,
        "alpha": 0.0 if no_recon else float(cfg.get("alpha", 0.5)),
        "beta": float(cfg.get("beta", 1e-2)),
        "use_ent": bool(cfg.get("use_ent", True)),
        "mixed_layout": mixed_layout,
        "mixed_numeric_scale": float(cfg.get("mixed_numeric_scale", 1.0)),
        "mixed_categorical_scale": float(cfg.get("mixed_categorical_scale", 1.0)),
    }


@torch.no_grad()
def _evaluate(model, loader, device, loss_kwargs) -> tuple[dict, np.ndarray, np.ndarray, list[str]]:
    model.eval()
    ys, scores, ids = [], [], []
    sums = {"total": 0.0, "cls": 0.0, "recon": 0.0, "ent": 0.0}
    n = 0
    for batch in loader:
        x, y = batch["x"].to(device), batch["y"].to(device)
        out = model(x, return_aux=True)
        loss, parts = total_loss(
            out["logits"], y, x, out["xhat"], out["z"], **loss_kwargs
        )
        bs = len(y)
        sums["total"] += float(loss) * bs
        for key in ("cls", "recon", "ent"):
            sums[key] += parts[key] * bs
        n += bs
        ys.append(y.cpu().numpy())
        scores.append(torch.softmax(out["logits"], dim=1)[:, 1].cpu().numpy())
        ids.extend(batch["record_id"])
    y_arr, score_arr = np.concatenate(ys), np.concatenate(scores)
    threshold = best_f1_threshold(y_arr, score_arr)
    metrics = binary_metrics(y_arr, score_arr, threshold)
    metrics.update({f"loss_{key}": value / max(1, n) for key, value in sums.items()})
    return metrics, y_arr, score_arr, ids


def _prediction_frame(ids, y, scores, threshold) -> pd.DataFrame:
    scores = np.asarray(scores, dtype=float)
    return pd.DataFrame(
        {
            "record_id": ids,
            "y_true": np.asarray(y, dtype=int),
            "y_score": scores,
            "y_pred": (scores >= threshold).astype(int),
            "threshold": float(threshold),
        }
    )


def _checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    epoch: int,
    val_metrics: dict,
    cfg: dict,
    data: PreparedData,
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "val_auc": val_metrics["auc"],
        "threshold": val_metrics["threshold"],
        "cfg": cfg,
        "preprocessing": data.preprocessing,
        "split_path": str(data.split_path),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _prepare_output(cfg: dict, smoke: bool, overwrite: bool) -> Path:
    value = cfg.get(
        "output_dir",
        cfg.get("ckpt_dir", f"checkpoints/asae_seed{int(cfg.get('seed', 0))}"),
    )
    out_dir = _resolve_path(value, ROOT)
    if smoke:
        out_dir = out_dir.with_name(out_dir.name + "_smoke")
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"output directory is not empty: {out_dir}; choose --output-dir or pass --overwrite"
            )
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def train_one(
    cfg: dict,
    smoke: bool = False,
    overwrite: bool = False,
    output_dir: str | Path | None = None,
) -> dict:
    cfg = dict(cfg)
    if output_dir is not None:
        cfg["output_dir"] = str(output_dir)
    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    torch.use_deterministic_algorithms(
        bool(cfg.get("deterministic_algorithms", True)), warn_only=True
    )
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = _prepare_output(cfg, smoke, overwrite)

    handlers: list[logging.Handler] = [
        logging.StreamHandler(),
        logging.FileHandler(out_dir / "train.log", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    data = prepare_data(cfg)
    d_in = int(data.train.x.shape[1])
    if "d_in" in cfg and int(cfg["d_in"]) != d_in:
        raise ValueError(f"configured d_in={cfg['d_in']} but prepared data has {d_in} columns")
    cfg["d_in"] = d_in
    model = build_model(cfg, d_in=d_in).to(device)
    LOG.info("%s", param_line(model))
    LOG.info(
        "device=%s seed=%d split=%s train=%d val=%d test=%d smote=%s class_weight=%s",
        device,
        seed,
        data.split_path,
        len(data.train),
        len(data.val),
        len(data.test),
        _sampling_requested(cfg),
        cfg.get("class_weight", False),
    )

    batch_size = int(cfg.get("smoke_batch_size", 32) if smoke else cfg.get("batch_size", 256))
    generator = torch.Generator().manual_seed(seed)
    loader_args = {
        "batch_size": batch_size,
        "num_workers": int(cfg.get("num_workers", 0)),
        "collate_fn": _collate,
        "pin_memory": bool(cfg.get("pin_memory", device.type == "cuda")),
    }
    train_loader = DataLoader(
        data.train, shuffle=True, generator=generator, **loader_args
    )
    val_loader = DataLoader(data.val, shuffle=False, **loader_args)
    test_loader = DataLoader(data.test, shuffle=False, **loader_args)

    w0, w1 = _class_weights(cfg.get("class_weight", False), data.original_train_y)
    loss_kwargs = _loss_kwargs(cfg, data.mixed_layout)
    loss_kwargs.update({"w0": w0, "w1": w1})
    LOG.info(
        "loss weights=(%.6f, %.6f) alpha=%.6f beta=%.6f no_recon=%s mixed_scales=(%.3f, %.3f)",
        w0,
        w1,
        loss_kwargs["alpha"],
        loss_kwargs["beta"],
        bool(cfg.get("no_recon", False)),
        loss_kwargs["mixed_numeric_scale"],
        loss_kwargs["mixed_categorical_scale"],
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg.get("lr", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
        betas=tuple(float(v) for v in cfg.get("adam_betas", [0.9, 0.999])),
    )
    epochs = int(cfg.get("smoke_epochs", 2) if smoke else cfg.get("max_epochs", 200))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=float(cfg.get("min_lr", 1e-6)),
    )
    early_stop = bool(cfg.get("early_stop", False))
    patience = int(cfg.get("patience", 20))
    min_epochs = int(cfg.get("min_epochs", 1))
    min_delta = float(cfg.get("min_delta", 0.0))

    fields = [
        "epoch",
        "lr",
        "train_loss",
        "train_cls",
        "train_recon",
        "train_ent",
        "val_loss",
        "val_cls",
        "val_recon",
        "val_ent",
        "val_accuracy",
        "val_precision",
        "val_recall",
        "val_f1",
        "val_auc",
        "val_ap",
        "val_brier",
        "val_threshold",
        "best_val_auc",
        "best_epoch",
        "bad_epochs",
        "epoch_seconds",
        "elapsed_seconds",
        "train_records",
        "learning_rate_next",
    ]
    history_path = out_dir / "history.csv"
    history_file = history_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(history_file, fieldnames=fields)
    writer.writeheader()

    best_auc, best_epoch, bad_epochs = -np.inf, 0, 0
    started = time.perf_counter()
    try:
        for epoch in range(1, epochs + 1):
            epoch_started = time.perf_counter()
            model.train()
            sums = {"total": 0.0, "cls": 0.0, "recon": 0.0, "ent": 0.0}
            seen = 0
            lr = float(optimizer.param_groups[0]["lr"])
            for batch in train_loader:
                x, y = batch["x"].to(device), batch["y"].to(device)
                optimizer.zero_grad(set_to_none=True)
                out = model(x, return_aux=True)
                loss, parts = total_loss(
                    out["logits"], y, x, out["xhat"], out["z"], **loss_kwargs
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at epoch {epoch}")
                loss.backward()
                clip = float(cfg.get("grad_clip", 0.0))
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                bs = len(y)
                seen += bs
                sums["total"] += float(loss.detach()) * bs
                for key in ("cls", "recon", "ent"):
                    sums[key] += parts[key] * bs

            val, _, _, _ = _evaluate(model, val_loader, device, loss_kwargs)
            improved = val["auc"] > best_auc + min_delta
            if improved:
                best_auc, best_epoch, bad_epochs = val["auc"], epoch, 0
                _checkpoint(
                    out_dir / "best.pt",
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    val,
                    cfg,
                    data,
                )
            else:
                bad_epochs += 1
            _checkpoint(
                out_dir / "last.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                val,
                cfg,
                data,
            )
            scheduler.step()
            epoch_seconds = time.perf_counter() - epoch_started
            row = {
                "epoch": epoch,
                "lr": lr,
                "train_loss": sums["total"] / max(1, seen),
                "train_cls": sums["cls"] / max(1, seen),
                "train_recon": sums["recon"] / max(1, seen),
                "train_ent": sums["ent"] / max(1, seen),
                "val_loss": val["loss_total"],
                "val_cls": val["loss_cls"],
                "val_recon": val["loss_recon"],
                "val_ent": val["loss_ent"],
                "val_accuracy": val["accuracy"],
                "val_precision": val["precision"],
                "val_recall": val["recall"],
                "val_f1": val["f1"],
                "val_auc": val["auc"],
                "val_ap": val["ap"],
                "val_brier": val["brier"],
                "val_threshold": val["threshold"],
                "best_val_auc": best_auc,
                "best_epoch": best_epoch,
                "bad_epochs": bad_epochs,
                "epoch_seconds": epoch_seconds,
                "elapsed_seconds": time.perf_counter() - started,
                "train_records": seen,
                "learning_rate_next": optimizer.param_groups[0]["lr"],
            }
            writer.writerow(row)
            history_file.flush()
            LOG.info(
                "epoch=%03d/%03d lr=%.7f train=%.5f (cls=%.5f rec=%.5f ent=%.5f) "
                "val_loss=%.5f val_auc=%.4f val_f1=%.4f threshold=%.4f best=%03d %.4f time=%.1fs",
                epoch,
                epochs,
                lr,
                row["train_loss"],
                row["train_cls"],
                row["train_recon"],
                row["train_ent"],
                row["val_loss"],
                row["val_auc"],
                row["val_f1"],
                row["val_threshold"],
                best_epoch,
                best_auc,
                epoch_seconds,
            )
            if early_stop and epoch >= min_epochs and bad_epochs >= patience:
                LOG.info("early stopping at epoch %d; best epoch %d", epoch, best_epoch)
                break
    finally:
        history_file.close()

    best = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(best["model"])
    val_metrics, yv, pv, val_ids = _evaluate(model, val_loader, device, loss_kwargs)
    threshold = float(best["threshold"])
    test_metrics, yt, pt, test_ids = _evaluate(model, test_loader, device, loss_kwargs)
    test_metrics = binary_metrics(yt, pt, threshold)
    _prediction_frame(val_ids, yv, pv, threshold).to_csv(
        out_dir / "predictions_val.csv", index=False
    )
    test_frame = _prediction_frame(test_ids, yt, pt, threshold)
    test_frame.to_csv(out_dir / "predictions_test.csv", index=False)
    prediction_dir = cfg.get("prediction_dir")
    if prediction_dir:
        pred_dir = _resolve_path(prediction_dir, ROOT)
        pred_dir.mkdir(parents=True, exist_ok=True)
        test_frame.to_csv(pred_dir / f"asae_seed{seed}.csv", index=False)

    result = {
        **test_metrics,
        "best_epoch": int(best_epoch),
        "best_val_auc": float(best_auc),
        "best_val_f1": float(val_metrics["f1"]),
        "epochs_completed": int(pd.read_csv(history_path).shape[0]),
        "seconds": float(time.perf_counter() - started),
        "output_dir": str(out_dir),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    LOG.info("test metrics: %s", json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ASAE on a named COIL 2000 split")
    parser.add_argument("--config", default=str(ROOT / "configs" / "asae.yaml"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--split-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="explicit two-epoch integration run; normal training is unchanged",
    )
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.split_path is not None:
        cfg["split_path"] = args.split_path
    result = train_one(
        cfg,
        smoke=args.smoke,
        overwrite=args.overwrite,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
