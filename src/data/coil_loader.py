from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import ROOT

COIL_DIR = ROOT / "data" / "coil2000"
CACHE = ROOT / "data" / "features" / "coil_table.npz"

N_FEATURES = 85
N_EXPECTED = 9822


def _read_whitespace_table(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append([float(tok) for tok in line.split()])
    return np.asarray(rows, dtype=np.float32)


def load_official_files(coil_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Join UCI ticdata + ticeval/tictgts on row order, then concatenate."""
    coil_dir = Path(coil_dir or COIL_DIR)
    train = _read_whitespace_table(coil_dir / "ticdata2000.txt")
    if train.shape[1] != 86:
        raise ValueError(f"ticdata2000.txt expected 86 columns, got {train.shape[1]}")
    eval_x = _read_whitespace_table(coil_dir / "ticeval2000.txt")
    eval_y = _read_whitespace_table(coil_dir / "tictgts2000.txt").reshape(-1)
    if len(eval_x) != len(eval_y):
        raise ValueError("ticeval2000.txt and tictgts2000.txt length mismatch")
    eval_xy = np.concatenate([eval_x, eval_y[:, None]], axis=1)
    table = np.concatenate([train, eval_xy], axis=0)
    x = table[:, :N_FEATURES].astype(np.float32)
    y = table[:, -1].astype(np.int64)
    return x, y


def load_coil_table(prefer_official: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y, record_ids). Official UCI files are used when present."""
    official = [
        COIL_DIR / "ticdata2000.txt",
        COIL_DIR / "ticeval2000.txt",
        COIL_DIR / "tictgts2000.txt",
    ]
    if prefer_official and all(p.exists() for p in official):
        x, y = load_official_files(COIL_DIR)
    elif CACHE.exists():
        blob = np.load(CACHE, allow_pickle=True)
        x = blob["x"].astype(np.float32)
        y = blob["y"].astype(np.int64)
    else:
        raise FileNotFoundError(
            "Neither official COIL files in data/coil2000/ nor data/features/coil_table.npz found."
        )
    if x.shape != (N_EXPECTED, N_FEATURES):
        raise ValueError(f"expected X shape {(N_EXPECTED, N_FEATURES)}, got {x.shape}")
    if y.shape != (N_EXPECTED,):
        raise ValueError(f"expected y length {N_EXPECTED}, got {y.shape}")
    ids = np.array([f"R{i:05d}" for i in range(len(y))])
    return x, y, ids


def write_manifest(path: Path, ids: np.ndarray, y: np.ndarray, split: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"record_id": ids, "y": y.astype(int), "split": split}).to_csv(path, index=False)
