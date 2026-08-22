from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: str | Path) -> dict:
    import yaml

    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_params(model) -> dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {
        "trainable": int(trainable),
        "frozen_est": int(frozen),
        "total_est": int(trainable + frozen),
    }


def param_line(model) -> str:
    c = count_params(model)
    return (
        f"Trainable {c['trainable']:,} | frozen_backbone_est {c['frozen_est']:,} "
        f"| total_est {c['total_est']:,}"
    )


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")
