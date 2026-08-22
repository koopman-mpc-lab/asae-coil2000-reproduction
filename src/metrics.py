from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_score)),
        "ap": float(average_precision_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
    }


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    thresholds = np.unique(np.concatenate(([0.0], y_score, [1.0])))
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_f1, best_t = score, float(t)
    return best_t


def roc_points(y_true: np.ndarray, y_score: np.ndarray):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return fpr, tpr


def participation_ratio(z: np.ndarray) -> float:
    z = np.asarray(z, dtype=np.float64)
    z = z - z.mean(axis=0, keepdims=True)
    cov = np.cov(z, rowvar=False)
    eig = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
    num = float(eig.sum() ** 2)
    den = float(np.square(eig).sum()) + 1e-12
    return num / den
