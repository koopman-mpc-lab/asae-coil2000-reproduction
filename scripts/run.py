from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COMMANDS = {
    "splits": [sys.executable, str(ROOT / "scripts" / "generate_splits.py")],
    "asae": [sys.executable, "-m", "src.train"],
    "baselines": [sys.executable, str(ROOT / "scripts" / "train_baselines.py")],
    "ablation": [sys.executable, str(ROOT / "scripts" / "run_ablations.py")],
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified entry point for splits, training, and ablations"
    )
    parser.add_argument("stage", choices=COMMANDS)
    args, remainder = parser.parse_known_args()
    subprocess.run(COMMANDS[args.stage] + remainder, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
