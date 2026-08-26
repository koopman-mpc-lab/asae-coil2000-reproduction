"""Type-consistent representation for the COIL mixed-type rerun.

Per the UCI data dictionary two attributes are nominal: MOSTYPE
(customer subtype, L0) and MOSHOOFD (customer main type, L2). The
remaining 83 attributes are ordinal bins, counts, or discretised
monetary levels and stay numeric. Nominal categories and numeric
standardisation statistics are fitted on the training partition only.

Oversampling for this representation uses SMOTENC on the raw integer
frame (categorical indices below), so added rows carry observed category
codes instead of fractional interpolations; one-hot expansion
is applied afterwards.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..utils import ROOT

NOMINAL_IDX = (0, 4)  # 0-based attribute index: MOSTYPE, MOSHOOFD
NOMINAL_NAMES = {0: "MOSTYPE", 4: "MOSHOOFD"}
MIXED_CACHE = ROOT / "data" / "features" / "x_mixed.npz"


def fit_mixed_spec(x: np.ndarray, train_idx: np.ndarray) -> dict:
    """Fit categories and numeric scaler on the training partition."""
    x_tr = x[train_idx]
    numeric_cols = np.array([j for j in range(x.shape[1]) if j not in NOMINAL_IDX])
    mu = x_tr[:, numeric_cols].mean(axis=0)
    sd = x_tr[:, numeric_cols].std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    categories = {j: np.unique(x_tr[:, j].astype(np.int64)) for j in NOMINAL_IDX}
    return {
        "numeric_cols": numeric_cols,
        "mu": mu.astype(np.float32),
        "sd": sd.astype(np.float32),
        "categories": categories,
    }


def transform_mixed(x: np.ndarray, spec: dict) -> np.ndarray:
    """Standardised numeric block followed by one-hot blocks (train categories).

    Categories unseen in the training partition map to an all-zero row of
    the corresponding block.
    """
    num = (x[:, spec["numeric_cols"]] - spec["mu"]) / spec["sd"]
    blocks = [num.astype(np.float32)]
    for j in NOMINAL_IDX:
        cats = spec["categories"][j]
        lookup = {int(c): k for k, c in enumerate(cats)}
        onehot = np.zeros((len(x), len(cats)), dtype=np.float32)
        for i, v in enumerate(x[:, j].astype(np.int64)):
            k = lookup.get(int(v))
            if k is not None:
                onehot[i, k] = 1.0
        blocks.append(onehot)
    return np.concatenate(blocks, axis=1)


def block_layout(spec: dict) -> tuple[int, list[int]]:
    """(numeric_dim, one-hot block sizes) in transform order."""
    return len(spec["numeric_cols"]), [len(spec["categories"][j]) for j in NOMINAL_IDX]


def smotenc_expand(
    x_raw: np.ndarray, y: np.ndarray, ratio: float = 1.0 / 3.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """SMOTENC on the raw integer frame; nominal columns use neighbour votes."""
    from imblearn.over_sampling import SMOTENC

    sm = SMOTENC(
        categorical_features=list(NOMINAL_IDX),
        sampling_strategy=ratio,
        random_state=seed,
        k_neighbors=5,
    )
    x_res, y_res = sm.fit_resample(x_raw, y)
    return np.asarray(x_res, dtype=np.float32), np.asarray(y_res, dtype=np.int64)


def build_mixed_cache(
    x: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
    train_idx: np.ndarray,
    out_path: Path | None = None,
) -> dict:
    """Write data/features/x_mixed.npz with the fitted layout metadata."""
    out_path = Path(out_path or MIXED_CACHE)
    spec = fit_mixed_spec(x, train_idx)
    x_mixed = transform_mixed(x, spec)
    numeric_dim, blocks = block_layout(spec)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        x=x_mixed,
        y=y.astype(np.int64),
        ids=ids,
        numeric_dim=np.int64(numeric_dim),
        block_sizes=np.asarray(blocks, dtype=np.int64),
        nominal_idx=np.asarray(NOMINAL_IDX, dtype=np.int64),
        mostype_categories=spec["categories"][0],
        moshoofd_categories=spec["categories"][4],
    )
    return {"d_in": x_mixed.shape[1], "numeric_dim": numeric_dim, "block_sizes": blocks}


def load_mixed_layout(path: Path | None = None) -> tuple[int, list[int]]:
    blob = np.load(Path(path or MIXED_CACHE), allow_pickle=True)
    return int(blob["numeric_dim"]), [int(b) for b in blob["block_sizes"]]
