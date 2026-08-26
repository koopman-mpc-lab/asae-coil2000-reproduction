from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from torch.utils.data import DataLoader

from .losses import recon_mse, weighted_ce
from .metrics import best_f1_threshold, binary_metrics
from .models import MLPClassifier, SymmetricAE
from .train import PreparedData, _collate, prepare_data
from .utils import ROOT, set_seed

MODEL_NAMES = (
    "lr",
    "svm",
    "rf",
    "xgboost",
    "lightgbm",
    "mlp",
    "ae",
    "dae",
    "vae",
    "sae",
    "rae",
    "tabnet",
)
NEURAL_NAMES = {"mlp", "ae", "dae", "vae", "sae", "rae"}


def _normalise_name(name: str) -> str:
    aliases = {
        "ae_mlp": "ae",
        "dae_mlp": "dae",
        "vae_mlp": "vae",
        "rae_mlp": "rae",
    }
    name = aliases.get(name.lower(), name.lower())
    if name not in MODEL_NAMES:
        raise ValueError(f"unknown baseline {name!r}; choose from {MODEL_NAMES}")
    return name


def _class_weight(enabled: bool) -> str | None:
    return "balanced" if enabled else None


def _build_classical(name: str, params: dict, seed: int, class_weight: bool):
    cw = _class_weight(class_weight)
    if name == "lr":
        return LogisticRegression(
            C=float(params.get("C", 1.0)),
            class_weight=cw,
            max_iter=int(params.get("max_iter", 3000)),
            random_state=seed,
            solver="liblinear",
        )
    if name == "svm":
        return SVC(
            C=float(params.get("C", 1.0)),
            gamma=params.get("gamma", "scale"),
            class_weight=cw,
            probability=True,
            random_state=seed,
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 500)),
            max_depth=params.get("max_depth"),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            max_features=params.get("max_features", "sqrt"),
            class_weight=cw,
            n_jobs=int(params.get("n_jobs", -1)),
            random_state=seed,
        )
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError("XGBoost selected; install xgboost") from exc
        return XGBClassifier(
            learning_rate=float(params.get("learning_rate", 0.05)),
            max_depth=int(params.get("max_depth", 5)),
            n_estimators=int(params.get("n_estimators", 400)),
            subsample=float(params.get("subsample", 0.8)),
            colsample_bytree=float(params.get("colsample_bytree", 0.8)),
            min_child_weight=float(params.get("min_child_weight", 1.0)),
            eval_metric="auc",
            n_jobs=int(params.get("n_jobs", -1)),
            random_state=seed,
        )
    if name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise RuntimeError("LightGBM selected; install lightgbm") from exc
        return LGBMClassifier(
            learning_rate=float(params.get("learning_rate", 0.05)),
            max_depth=int(params.get("max_depth", -1)),
            n_estimators=int(params.get("n_estimators", 400)),
            subsample=float(params.get("subsample", 0.8)),
            colsample_bytree=float(params.get("colsample_bytree", 0.8)),
            min_child_weight=float(params.get("min_child_weight", 1.0)),
            class_weight=cw,
            n_jobs=int(params.get("n_jobs", -1)),
            random_state=seed,
            verbosity=-1,
        )
    raise ValueError(f"{name} is not a classical estimator")


def _fit_classical(name: str, params: dict, data: PreparedData, seed: int, cw: bool):
    estimator = _build_classical(name, params, seed, cw)
    estimator.fit(data.train.x, data.train.y)
    return estimator


def _score_estimator(estimator, x: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return np.asarray(estimator.predict_proba(x)[:, 1], dtype=float)
    return np.asarray(estimator.decision_function(x), dtype=float)


def _build_torch(name: str, d_in: int, params: dict):
    if name == "mlp":
        return MLPClassifier(
            d_in=d_in,
            dropout=float(params.get("dropout", 0.3)),
        )
    return SymmetricAE(
        d_in=d_in,
        d_z=int(params.get("d_z", 32)),
        dropout=float(params.get("dropout", 0.3)),
        noise_std=float(params.get("noise_std", 0.18 if name == "dae" else 0.0)),
        latent_l2=float(params.get("latent_l2", 0.006 if name == "rae" else 0.0)),
        variational=name == "vae",
    )


@torch.no_grad()
def _torch_scores(model, dataset, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    values = []
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate
    )
    for batch in loader:
        logits = model(batch["x"].to(device), return_aux=True)["logits"]
        values.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(values)


def _fit_torch(
    name: str,
    params: dict,
    data: PreparedData,
    seed: int,
    cfg: dict,
):
    set_seed(seed)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = _build_torch(name, data.train.x.shape[1], params).to(device)
    batch_size = int(params.get("batch_size", cfg.get("batch_size", 256)))
    loader = DataLoader(
        data.train,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=_collate,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params.get("lr", 8e-4)),
        weight_decay=float(params.get("weight_decay", 1e-4)),
    )
    max_epochs = int(params.get("max_epochs", cfg.get("baseline_epochs", 200)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, max_epochs), eta_min=float(cfg.get("min_lr", 1e-6))
    )
    n0, n1 = max(1, int((data.original_train_y == 0).sum())), max(
        1, int((data.original_train_y == 1).sum())
    )
    if bool(cfg.get("class_weight", True)):
        n = n0 + n1
        w0, w1 = 0.5 * n / n0, 0.5 * n / n1
    else:
        w0 = w1 = 1.0
    best_auc, best_state, bad = -np.inf, None, 0
    patience = int(cfg.get("baseline_patience", 25))
    for _epoch in range(max_epochs):
        model.train()
        for batch in loader:
            x, y = batch["x"].to(device), batch["y"].to(device)
            out = model(x, return_aux=True)
            loss = weighted_ce(out["logits"], y, w0, w1)
            if name != "mlp":
                recon_weight = float(
                    params.get("recon_weight", 0.7 if name == "sae" else 0.5)
                )
                loss = loss + recon_weight * recon_mse(x, out["xhat"])
                if name == "vae":
                    logvar = out["logvar"]
                    mu = out["z"]
                    kl = -0.5 * (1.0 + logvar - mu.square() - logvar.exp()).sum(1).mean()
                    loss = loss + float(params.get("kl_weight", 0.08)) * kl
                if name == "rae":
                    loss = loss + float(params.get("latent_l2", 0.006)) * out["z"].square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()
        val_score = _torch_scores(model, data.val, device, batch_size)
        val_auc = float(roc_auc_score(data.val.y, val_score))
        if val_auc > best_auc + 1e-5:
            best_auc = val_auc
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is None:
        raise RuntimeError("neural baseline did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    return model, device, batch_size


def _fit_tabnet(params: dict, data: PreparedData, seed: int, cfg: dict):
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
    except ImportError as exc:
        raise RuntimeError("TabNet selected; install pytorch-tabnet") from exc
    classifier = TabNetClassifier(
        n_d=int(params.get("n_d", 24)),
        n_a=int(params.get("n_a", 24)),
        n_steps=int(params.get("n_steps", 5)),
        gamma=float(params.get("gamma", 1.3)),
        lambda_sparse=float(params.get("lambda_sparse", 8e-4)),
        optimizer_fn=torch.optim.AdamW,
        optimizer_params={
            "lr": float(params.get("lr", 0.012)),
            "weight_decay": float(params.get("weight_decay", 1e-5)),
        },
        scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingLR,
        scheduler_params={
            "T_max": int(cfg.get("baseline_epochs", 200)),
            "eta_min": float(cfg.get("min_lr", 1e-6)),
        },
        seed=seed,
        verbose=int(cfg.get("tabnet_verbose", 10)),
    )
    classifier.fit(
        data.train.x,
        data.train.y,
        eval_set=[(data.val.x, data.val.y)],
        eval_name=["val"],
        eval_metric=["auc"],
        max_epochs=int(cfg.get("baseline_epochs", 200)),
        patience=int(cfg.get("baseline_patience", 25)),
        batch_size=int(cfg.get("tabnet_batch_size", 1024)),
        virtual_batch_size=int(cfg.get("tabnet_virtual_batch_size", 128)),
        weights=1 if bool(cfg.get("class_weight", True)) else 0,
        drop_last=False,
    )
    return classifier


def _trial_params(trial, name: str) -> dict:
    if name == "lr":
        return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True)}
    if name == "svm":
        return {
            "C": trial.suggest_float("C", 1e-2, 1e2, log=True),
            "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True),
        }
    if name == "rf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 700, step=100),
            "max_depth": trial.suggest_int("max_depth", 5, 24),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 12),
            "max_features": trial.suggest_float("max_features", 0.2, 0.9),
        }
    if name in {"xgboost", "lightgbm"}:
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "n_estimators": trial.suggest_int("n_estimators", 150, 650, step=50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 8.0),
        }
    if name == "tabnet":
        return {
            "n_d": trial.suggest_categorical("n_d", [8, 16, 24, 32]),
            "n_a": trial.suggest_categorical("n_a", [8, 16, 24, 32]),
            "n_steps": trial.suggest_int("n_steps", 3, 7),
            "gamma": trial.suggest_float("gamma", 1.0, 1.8),
            "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-6, 1e-2, log=True),
            "lr": trial.suggest_float("lr", 1e-3, 3e-2, log=True),
        }
    params = {
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "d_z": trial.suggest_categorical("d_z", [16, 32, 48, 64]),
    }
    if name == "dae":
        params["noise_std"] = trial.suggest_float("noise_std", 0.05, 0.35)
    if name == "vae":
        params["kl_weight"] = trial.suggest_float("kl_weight", 1e-3, 0.2, log=True)
    if name in {"ae", "sae"}:
        params["recon_weight"] = trial.suggest_float("recon_weight", 0.1, 1.5)
    if name == "rae":
        params["latent_l2"] = trial.suggest_float("latent_l2", 1e-4, 5e-2, log=True)
    return params


def tune(
    name: str,
    data: PreparedData,
    cfg: dict,
    trials: int,
    study_path: Path,
) -> dict:
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Optuna tuning requested; install optuna") from exc
    name = _normalise_name(name)
    seed = int(cfg.get("seed", 0))
    tune_cfg = dict(cfg)
    tune_cfg["baseline_epochs"] = int(cfg.get("tune_epochs", 60))
    tune_cfg["baseline_patience"] = int(cfg.get("tune_patience", 12))

    def objective(trial) -> float:
        params = _trial_params(trial, name)
        if name in NEURAL_NAMES:
            model, device, batch_size = _fit_torch(name, params, data, seed, tune_cfg)
            score = _torch_scores(model, data.val, device, batch_size)
        elif name == "tabnet":
            model = _fit_tabnet(params, data, seed, tune_cfg)
            score = model.predict_proba(data.val.x)[:, 1]
        else:
            model = _fit_classical(
                name, params, data, seed, bool(cfg.get("class_weight", True))
            )
            score = _score_estimator(model, data.val.x)
        return float(roc_auc_score(data.val.y, score))

    study = optuna.create_study(
        study_name=f"{name}_seed{seed}",
        direction="maximize",
        storage=f"sqlite:///{study_path.as_posix()}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    remaining = max(0, int(trials) - len(study.trials))
    if remaining:
        study.optimize(objective, n_trials=remaining)
    pd.DataFrame(
        [
            {
                "number": trial.number,
                "value": trial.value,
                "state": str(trial.state),
                **trial.params,
            }
            for trial in study.trials
        ]
    ).to_csv(study_path.with_suffix(".csv"), index=False)
    return dict(study.best_params)


def run_baseline(
    name: str,
    params: dict,
    data_cfg: dict,
    output_dir: str | Path,
    overwrite: bool = False,
    tune_trials: int = 0,
) -> dict:
    name = _normalise_name(name)
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{out_dir} is not empty; pass --overwrite")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = prepare_data(data_cfg)
    seed = int(data_cfg.get("seed", 0))
    started = time.perf_counter()
    if tune_trials:
        params = {
            **params,
            **tune(name, data, data_cfg, tune_trials, out_dir / "optuna.db"),
        }

    if name in NEURAL_NAMES:
        model, device, batch_size = _fit_torch(name, params, data, seed, data_cfg)
        val_score = _torch_scores(model, data.val, device, batch_size)
        test_score = _torch_scores(model, data.test, device, batch_size)
        torch.save(
            {"model": model.state_dict(), "name": name, "params": params, "cfg": data_cfg},
            out_dir / "model.pt",
        )
    elif name == "tabnet":
        model = _fit_tabnet(params, data, seed, data_cfg)
        val_score = model.predict_proba(data.val.x)[:, 1]
        test_score = model.predict_proba(data.test.x)[:, 1]
        model.save_model(str(out_dir / "tabnet"))
    else:
        model = _fit_classical(
            name, params, data, seed, bool(data_cfg.get("class_weight", True))
        )
        val_score = _score_estimator(model, data.val.x)
        test_score = _score_estimator(model, data.test.x)
        joblib.dump(model, out_dir / "model.joblib")

    threshold = best_f1_threshold(data.val.y, val_score)
    metrics = binary_metrics(data.test.y, test_score, threshold)
    frame = pd.DataFrame(
        {
            "record_id": data.test.ids,
            "y_true": data.test.y,
            "y_score": test_score,
            "y_pred": (test_score >= threshold).astype(int),
            "threshold": threshold,
        }
    )
    frame.to_csv(out_dir / "predictions_test.csv", index=False)
    summary = {
        "model": name,
        "seed": seed,
        "params": params,
        "val_auc": float(roc_auc_score(data.val.y, val_score)),
        "threshold": float(threshold),
        "seconds": float(time.perf_counter() - started),
        **metrics,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (out_dir / "config.json").write_text(
        json.dumps(data_cfg, indent=2), encoding="utf-8"
    )
    return summary
