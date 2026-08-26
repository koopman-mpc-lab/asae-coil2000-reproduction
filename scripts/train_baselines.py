from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baseline_pipeline import MODEL_NAMES, run_baseline
from src.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and tune COIL 2000 baselines")
    parser.add_argument("--config", default=str(ROOT / "configs" / "baselines.yaml"))
    parser.add_argument("--model", choices=["all", *MODEL_NAMES], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-path", default="primary.npz")
    parser.add_argument("--mixed", action="store_true")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--trials", type=int)
    parser.add_argument("--smote", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--class-weight", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--output-root", default="runs/baselines")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    selected_key = "selected_mixedtype" if args.mixed else "selected"
    selected = cfg[selected_key]
    data_cfg = {
        "seed": args.seed,
        "features": "x_mixed.npz" if args.mixed else "x_std.npz",
        "mixed_recon": args.mixed,
        "smotenc_categorical": [0, 4],
        "smote": args.smote,
        "smote_ratio": float(cfg.get("smote_ratio", 1.0 / 3.0)),
        "class_weight": args.class_weight,
        "split_path": args.split_path,
        "batch_size": int(cfg.get("batch_size", 256)),
        "baseline_epochs": int(cfg.get("baseline_epochs", 200)),
        "baseline_patience": int(cfg.get("baseline_patience", 25)),
        "tune_epochs": int(cfg.get("tune_epochs", 60)),
        "tune_patience": int(cfg.get("tune_patience", 12)),
        "min_lr": float(cfg.get("min_lr", 1e-6)),
        "tabnet_batch_size": int(cfg.get("tabnet_batch_size", 1024)),
        "tabnet_virtual_batch_size": int(
            cfg.get("tabnet_virtual_batch_size", 128)
        ),
    }
    models = MODEL_NAMES if args.model == "all" else (args.model,)
    trials = int(
        args.trials if args.trials is not None else cfg.get("optuna_trials", 50)
    )
    failures = []
    for model in models:
        params = dict(selected.get(model, {}))
        output = Path(args.output_root) / (
            f"{model}_{'mixed_' if args.mixed else ''}seed{args.seed}"
        )
        try:
            result = run_baseline(
                model,
                params,
                data_cfg,
                output,
                overwrite=args.overwrite,
                tune_trials=trials if args.tune else 0,
            )
            print(json.dumps(result, indent=2))
        except (ImportError, RuntimeError) as exc:
            failures.append((model, str(exc)))
            print(f"{model}: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                raise SystemExit(2) from exc
    if failures:
        print("Unavailable baselines:", file=sys.stderr)
        for model, reason in failures:
            print(f"  {model}: {reason}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
