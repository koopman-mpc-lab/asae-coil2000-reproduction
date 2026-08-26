from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.coil_loader import load_coil_table


def _primary(y: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    rng.shuffle(pos)
    rng.shuffle(neg)
    train = np.concatenate([pos[:411], neg[:6465]])
    val = np.concatenate([pos[411:499], neg[6465:7850]])
    test = np.concatenate([pos[499:586], neg[7850:9236]])
    return {
        "train_idx": np.sort(train),
        "val_idx": np.sort(val),
        "test_idx": np.sort(test),
    }


def _official(y: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    official_train = np.arange(5822)
    official_test = np.arange(5822, len(y))
    train_idx, val_idx = train_test_split(
        official_train,
        test_size=0.20,
        random_state=seed,
        stratify=y[official_train],
    )
    return {"train_idx": train_idx, "val_idx": val_idx, "test_idx": official_test}


def _group_ids(x: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(x)
    _, inverse = np.unique(contiguous, axis=0, return_inverse=True)
    return inverse


def _best_group_holdout(
    indices: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    fraction: float,
    seed: int,
    attempts: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    target_n = fraction * len(indices)
    target_rate = float(y[indices].mean())
    splitter = GroupShuffleSplit(
        n_splits=attempts, test_size=fraction, random_state=seed
    )
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for kept_local, held_local in splitter.split(
        indices, y[indices], groups=groups[indices]
    ):
        kept, held = indices[kept_local], indices[held_local]
        if len(np.unique(y[held])) < 2 or len(np.unique(y[kept])) < 2:
            continue
        size_error = abs(len(held) - target_n) / max(1.0, target_n)
        rate_error = abs(float(y[held].mean()) - target_rate) / max(target_rate, 1e-8)
        score = size_error + 2.0 * rate_error
        if best is None or score < best[0]:
            best = (score, kept, held)
    if best is None:
        raise RuntimeError("could not find a grouped split containing both classes")
    return best[1], best[2]


def _grouped(
    x: np.ndarray, y: np.ndarray, columns: slice, seed: int
) -> dict[str, np.ndarray]:
    groups = _group_ids(x[:, columns])
    all_idx = np.arange(len(y))
    train_val, test_idx = _best_group_holdout(
        all_idx, y, groups, fraction=0.15, seed=seed
    )
    relative_val_fraction = 0.15 / 0.85
    train_idx, val_idx = _best_group_holdout(
        train_val,
        y,
        groups,
        fraction=relative_val_fraction,
        seed=seed + 1,
    )
    return {"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx}


def _validate(parts: dict[str, np.ndarray], n: int, groups: np.ndarray | None = None) -> None:
    named = [np.asarray(parts[f"{name}_idx"], dtype=int) for name in ("train", "val", "test")]
    joined = np.concatenate(named)
    if len(joined) != n or len(np.unique(joined)) != n:
        raise ValueError("split must cover every row exactly once")
    if joined.min() != 0 or joined.max() != n - 1:
        raise ValueError("split contains out-of-range indices")
    if groups is not None:
        group_sets = [set(groups[idx].tolist()) for idx in named]
        if any(group_sets[i] & group_sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("a group appears in more than one partition")


def _write(
    name: str,
    parts: dict[str, np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
    out_dir: Path,
    seed: int,
    overwrite: bool,
    group_columns: int | None = None,
) -> None:
    path = out_dir / f"{name}.npz"
    manifest = out_dir / f"{name}.csv"
    summary_path = out_dir / f"{name}.json"
    existing = [p for p in (path, manifest, summary_path) if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{existing[0]} exists; pass --overwrite to replace it")
    groups = _group_ids(x[:, :group_columns]) if group_columns is not None else None
    _validate(parts, len(y), groups=groups)
    np.savez_compressed(path, **{key: np.sort(value) for key, value in parts.items()})

    split_label = np.full(len(y), "", dtype=object)
    summary: dict[str, object] = {
        "protocol": name,
        "seed": seed,
        "group_columns": group_columns,
        "partitions": {},
    }
    for partition in ("train", "val", "test"):
        idx = parts[f"{partition}_idx"]
        split_label[idx] = partition
        summary["partitions"][partition] = {
            "n": int(len(idx)),
            "n_pos": int(y[idx].sum()),
            "positive_rate": float(y[idx].mean()),
        }
    pd.DataFrame(
        {
            "row_index": np.arange(len(y)),
            "record_id": ids,
            "y": y,
            "split": split_label,
        }
    ).to_csv(manifest, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{name}: {summary['partitions']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate COIL 2000 evaluation splits")
    parser.add_argument(
        "--protocol",
        choices=["all", "primary", "official", "group-exact", "group-demographic"],
        default="all",
    )
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    x, y, ids = load_coil_table()
    builders = {
        "primary": lambda: (_primary(y, args.seed), None),
        "official": lambda: (_official(y, args.seed), None),
        "group-exact": lambda: (
            _grouped(x, y, slice(0, 85), args.seed),
            85,
        ),
        "group-demographic": lambda: (
            _grouped(x, y, slice(0, 43), args.seed),
            43,
        ),
    }
    selected = builders if args.protocol == "all" else {args.protocol: builders[args.protocol]}
    for name, builder in selected.items():
        parts, group_columns = builder()
        _write(
            name.replace("-", "_"),
            parts,
            x,
            y,
            ids,
            args.out_dir,
            args.seed,
            args.overwrite,
            group_columns,
        )


if __name__ == "__main__":
    main()
