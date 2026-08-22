"""Build the type-consistent feature cache (data/features/x_mixed.npz)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.coil_loader import load_coil_table
from src.data.mixed import MIXED_CACHE, NOMINAL_NAMES, build_mixed_cache


def main():
    x, y, ids = load_coil_table()
    splits = np.load(ROOT / "data" / "splits" / "primary.npz", allow_pickle=True)
    info = build_mixed_cache(x, y, ids, splits["train_idx"].astype(int))
    print(f"wrote {MIXED_CACHE}")
    print(f"d_in={info['d_in']}  numeric_dim={info['numeric_dim']}")
    for (j, name), size in zip(sorted(NOMINAL_NAMES.items()), info["block_sizes"]):
        print(f"{name} (attr {j + 1}): {size} training categories")


if __name__ == "__main__":
    main()
