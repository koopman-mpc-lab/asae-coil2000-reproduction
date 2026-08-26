# ASAE on COIL 2000

Attention-based Symmetric AutoEncoder for caravan-insurance policy-ownership
classification on the UCI COIL 2000 benchmark.

## Install

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Usage

```bash
python scripts/run.py splits --protocol primary --seed 20260817
python scripts/run.py asae --config configs/asae.yaml --seed 2 --output-dir runs/asae_seed2
python scripts/run.py baselines --model all --seed 0 --output-root runs/baselines
python scripts/run.py ablation --kind variants --seeds 0 1 2 3 4 --output-root runs/ablations
```

`--protocol` also accepts `official`, `group-exact`, and `group-demographic`;
pass the matching `--split-path` to the training commands. Baselines are LR,
SVM, RF, XGBoost, LightGBM, MLP, AE, DAE, VAE, SAE, RAE, and TabNet, with
`--tune --trials 50` for the Optuna search. `configs/asae_mixed.yaml` selects
the mixed-type variant.

## Contents

```text
configs/      run settings          data/         UCI files and splits
src/          models and training   results/      predictions, metrics, curves, tables
scripts/      pipeline stages       checkpoints/  ASAE weights (seed 2)
```

Per-seed test scores, frozen thresholds, learning curves, latent vectors, and
attention gates are archived under `results/`. Weights and latents are kept for
seed 2; other seeds follow from the recorded configs, split indices, and seeds.
A single-record check:

```bash
python examples/infer_one_record.py
```

## Data

COIL 2000 is redistributed under its
[UCI terms](https://archive.ics.uci.edu/dataset/125/insurance+company+benchmark+coil+2000).
Code is released under the licence in `LICENSE`.
