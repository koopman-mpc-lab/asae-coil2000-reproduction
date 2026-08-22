from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils import ROOT
from .coil_loader import load_coil_table


class CoilDataset(Dataset):
    """Standardised COIL records for a named split."""

    def __init__(
        self,
        split: str = "test",
        features_path: str | Path | None = None,
        split_path: str | Path | None = None,
    ):
        features_path = Path(features_path or ROOT / "data" / "features" / "x_std.npz")
        split_path = Path(split_path or ROOT / "data" / "splits" / "primary.npz")
        if not features_path.exists():
            raise FileNotFoundError(f"missing {features_path}; run the data prep step first")
        blob = np.load(features_path, allow_pickle=True)
        splits = np.load(split_path, allow_pickle=True)
        key = {"train": "train_idx", "val": "val_idx", "test": "test_idx"}[split]
        idx = splits[key].astype(int)
        self.x = blob["x"][idx].astype(np.float32)
        self.y = blob["y"][idx].astype(np.int64)
        self.ids = np.asarray(blob["ids"])[idx]
        self.split = split
        self.idx = idx

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> dict:
        return {
            "x": torch.from_numpy(self.x[i]),
            "y": torch.tensor(int(self.y[i]), dtype=torch.long),
            "record_id": str(self.ids[i]),
        }


def collate_records(batch: list[dict]) -> dict:
    return {
        "x": torch.stack([b["x"] for b in batch], dim=0),
        "y": torch.stack([b["y"] for b in batch], dim=0),
        "record_id": [b["record_id"] for b in batch],
    }


def load_raw_if_needed():
    return load_coil_table()
