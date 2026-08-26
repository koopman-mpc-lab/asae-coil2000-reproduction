from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.train import train_one
from src.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ASAE component and latent-size studies")
    parser.add_argument("--base-config", default=str(ROOT / "configs" / "asae.yaml"))
    parser.add_argument("--ablation-config", default=str(ROOT / "configs" / "ablation.yaml"))
    parser.add_argument("--kind", choices=["all", "variants", "latent"], default="all")
    parser.add_argument("--variant", action="append")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--split-path", default="primary.npz")
    parser.add_argument("--output-root", default="runs/ablations")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    base = load_yaml(args.base_config)
    study = load_yaml(args.ablation_config)
    root = Path(args.output_root)
    if not root.is_absolute():
        root = ROOT / root
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "summary.csv"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"{summary_path} exists; pass --overwrite")

    jobs: list[tuple[str, dict]] = []
    if args.kind in {"all", "variants"}:
        variants = args.variant or list(study["variants"])
        unknown = set(variants) - set(study["variants"])
        if unknown:
            raise ValueError(f"unknown variants: {sorted(unknown)}")
        jobs.extend((name, dict(study["variants"][name])) for name in variants)
    if args.kind in {"all", "latent"}:
        jobs.extend(
            (f"dim{int(d_z)}", {"d_z": int(d_z)})
            for d_z in study["latent_dims"]
        )

    rows = []
    for label, overrides in jobs:
        for seed in args.seeds:
            cfg = {
                **base,
                **overrides,
                "seed": int(seed),
                "split_path": args.split_path,
            }
            output = root / f"{label}_seed{seed}"
            result = train_one(
                cfg,
                smoke=args.smoke,
                overwrite=args.overwrite,
                output_dir=output,
            )
            rows.append({"study": label, "seed": seed, **result})
            print(json.dumps(rows[-1], indent=2))
    pd.DataFrame(rows).to_csv(summary_path, index=False)


if __name__ == "__main__":
    main()
