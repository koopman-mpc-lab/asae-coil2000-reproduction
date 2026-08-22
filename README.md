# ASAE on COIL 2000

Training code and the archived files needed to recompute the tables
for the COIL 2000 caravan-insurance purchase task.

## What is released

- `src/`, `configs/`: ASAE, baselines, official COIL loader
- `data/coil2000/`: `ticdata2000.txt`, `ticeval2000.txt`, `tictgts2000.txt`
- `data/splits/`: primary 70/15/15 indices (411 / 88 / 87 positives) and resplit seeds 10–14
- `results/predictions/`: one frozen `y_true_test.csv` plus per-seed scores for the main table, ablations, latent-dimension sweep, resplits, and the imbalance factorial
- `results/tables/`, `results/metrics/`: published means and the paired tests
- `results/curves/asae_seed*.csv`: ASAE epoch histories (seed 2 is the 200-epoch run restored at epoch 148)
- `logs/`: ASAE seeds 0–4, TabNet seed 2, the two named ablations, and sklearn/boosting search notes
- `scripts/check_table_consistency.py`: recomputes Table 3 from the archived predictions

Weights are not shipped. Table entries come from the archived prediction files.

## Quick checks

```bash
pip install -r requirements.txt
python scripts/smoke_forward.py
python scripts/check_table_consistency.py
python -m src.evaluate --pred results/predictions/main/asae_seed2.csv
```

`check_table_consistency.py` is also the CI job in `.github/workflows/tables.yml`.

## Frozen operating point (primary split, seeds 0–4)

| Model | Accuracy | F1 | AUC |
|-------|----------|----|-----|
| TabNet | 0.9483 | 0.5403 ± 0.0092 | 0.8261 ± 0.0055 |
| ASAE | 0.9586 | 0.6212 ± 0.0087 | 0.8641 ± 0.0048 |

Seeds are shared across models within a run. Thresholds were chosen on the
validation partition (max F1) and then frozen.

## Data

`src/data/coil_loader.py` joins `ticeval2000.txt` with `tictgts2000.txt` on
row order, concatenates the 5822 + 4000 official rows, then applies the
archived stratified indices. See `data/coil2000/README.txt`.
