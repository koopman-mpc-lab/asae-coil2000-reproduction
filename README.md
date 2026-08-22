# ASAE on COIL 2000

Code and archived runs for the Attention-based Symmetric AutoEncoder
experiments on the COIL 2000 caravan-insurance purchase task. Official UCI
extracts sit under `data/coil2000/`; standardised caches under `data/features/`
are enough to score a checkpoint and redraw the result figures.

## Layout

```
configs/          ASAE (85-d and 133-d mixed), baseline search, ablation flags
src/              model, losses, COIL loader, mixed-type prep, train / eval
data/
  coil2000/       official ticdata / ticeval / tictgts extracts
  splits/         primary 70/15/15 indices and resplit seeds 10–14
  features/       standardised table, x_mixed.npz, per-record test caches
  cases/          one test purchaser used in the attention walkthrough
results/          tables, per-seed predictions (main + mixedtype), curves
logs/YYYY-MM-DD/  training logs, pip_freeze, nvidia-smi
checkpoints/      ASAE seeds 0–4
.github/workflows table-check on the archived prediction files
scripts/          smoke, paced eval, figure redraw, table check, mixed cache
examples/         infer_one_record.py
figures/          redrawn from results/
```

## Environment

```bash
pip install -r requirements.txt
# optional
pip install -e .
```

PyTorch 2.1, scikit-learn 1.3, XGBoost 2.0, LightGBM 4.1, pytorch-tabnet 4.1.
The workstation used for the archived runs was an RTX 3090 (24 GB), i9-12900K,
64 GB RAM.

## Quick checks

```bash
python scripts/smoke_forward.py
python scripts/plot_from_results.py
python -m src.evaluate --pred results/predictions/main/asae_seed2.csv
python -m src.evaluate --pred results/predictions/mixedtype/asae_seed2.csv
python scripts/check_table_consistency.py
python scripts/eval_checkpoint.py --ckpt checkpoints/asae_seed2/best.pt --split test
python examples/infer_one_record.py
```

`eval_checkpoint.py` walks the cached test records with tile-style I/O +
forward pacing (`--pace-sec`, default 0.5 s/batch). `--fast` skips the wait.

`check_table_consistency.py` recomputes Table 3 (and the type-consistent
rerun table) from the archived per-seed prediction files and compares them
to `results/tables/tab3_main.csv` / `tab7_mixedtype.csv`. Every prediction
file shares the same frozen `results/predictions/y_true_test.csv`.

## Frozen operating point (primary split, five paired seeds)

| Model | Accuracy | F1 | AUC |
|-------|----------|----|-----|
| TabNet | 0.9483 | 0.5403 ± 0.0092 | 0.8261 ± 0.0055 |
| ASAE | 0.9586 | 0.6212 ± 0.0087 | 0.8641 ± 0.0048 |
| TabNet (133-d mixed) | 0.9489 | 0.5501 ± 0.0089 | 0.8342 ± 0.0077 |
| ASAE (133-d mixed) | 0.9591 | 0.6290 ± 0.0085 | 0.8706 ± 0.0047 |

Seeds `{0,1,2,3,4}` are shared across models within a run. Thresholds are
chosen on the validation partition (max F1) and then frozen. Seed 2 is the
run used for the confusion matrix and the one-record demo; its checkpoint
was restored from epoch 148 (validation AUC) after a 200-epoch monitoring
window.

Classical estimators have prediction dumps only. Neural runs also keep
epoch-wise histories under `results/curves/`.

## Data

`src/data/coil_loader.py` joins `ticeval2000.txt` with `tictgts2000.txt` on
row order, concatenates the 5822 + 4000 official rows, then applies the
archived stratified indices (411 / 88 / 87 positives). SMOTE (1:3) and
standardisation are fit on the training partition only. See
`data/coil2000/README.txt`.

For the type-consistent rerun, `scripts/build_mixed_features.py` rebuilds
`data/features/x_mixed.npz` (83 standardised numeric columns + one-hot
MOSTYPE / MOSHOOFD, 133 columns total; training-partition categories).
Training on that cache uses SMOTENC (cat = [0, 4], 1:3) and the mixed
reconstruction loss in `src/losses.py`; see `configs/asae_mixed.yaml` and
`configs/baselines.yaml:selected_mixedtype`. Dumps sit under
`results/predictions/mixedtype/`.

## Retrain

```bash
python -m src.train --config configs/asae.yaml --seed 2
```

This fits the released architecture on the cached standardised table. It is
a convenience entry point; the numbers in `results/tables/` come from the
archived prediction files, not from a fresh run of this script.
