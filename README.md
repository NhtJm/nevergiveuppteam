# Datathon 2026 — Task 3: Sales Forecasting Pipeline

End-to-end forecasting pipeline for daily `Revenue` and `COGS` over `2023-01-01 → 2024-07-01` (548 days), competition `datathon-2026-round-1` on Kaggle.

## Architecture

A four-layer ensemble combining gradient-boosted trees, linear regression, and per-quarter specialists, finished by a scalar calibration step.

```
sales.csv (2012-2022, log1p target, sample-weight 2014-2018)
    │
    ▼
[~88 features: calendar + Fourier + Tet windows + per-promo + real promo aggregates]
    │
    ▼
─── Layer 1 — 3 GBM boosters (each: 1 base + 4 quarterly specialists) ───
    LightGBM  ───┐
    XGBoost   ───┤── per-target weighted GBM_blend
    CatBoost  ───┘
    │
    ▼
─── Layer 2 — Ridge (z-score normalized features) ───
    │
    ▼
─── Layer 3 — Ensemble: 0.20 × Ridge + 0.80 × GBM_blend ───
    │
    ▼
─── Layer 4 — Calibration: × CR (Revenue), × CC (COGS) ───
    │
    ▼
submission_final.csv
```

### Why this works

1. **Sample weighting (2014-2018 emphasized 100×)** — the e-commerce business has three structural regimes (early-stage 2013, stable mid-period 2014-2018, COVID-shifted 2019-2022). Training primarily on the mid-period gives the cleanest signal; the calibration step compensates for the test-era trend.
2. **Three diverse boosters** — LightGBM (leaf-wise growth), XGBoost (level-wise), CatBoost (ordered boosting). Their errors are partially independent so blending reduces variance.
3. **Quarterly specialists** — each booster runs 5× (1 base + 4 quarter-boosted variants), with the matching quarter's specialist contributing 60% to the per-day prediction. Captures intra-year heterogeneity.
4. **Ridge regularizer** — small linear contribution stabilizes the long-horizon trend that tree models tend to flatten.
5. **Calibration multipliers** (CR=1.26 Revenue, CC=1.32 COGS) — both targets undershoot test-era values when trained on the mid-period; tuned scalar correction closes the gap.

## Project layout

```
.
├── data/                                    # 14 raw competition CSVs
│   ├── sales.csv, customers.csv, orders.csv, ...
│   └── sample_submission.csv
├── src/
│   ├── config.py                            # constants, paths, hyperparameters
│   ├── features.py                          # ~88 calendar/Fourier/Tet/promo features
│   ├── models.py                            # LGB + XGB + CatBoost + Ridge trainers
│   ├── ensemble.py                          # blending + calibration helpers
│   ├── train.py                             # main pipeline (orchestrator)
│   ├── components.py                        # alternative: Revenue = orders × AOV
│   ├── recursive.py                         # alternative: HGB + recursive forecasting
│   ├── tune.py                              # Optuna hyperparameter search
│   ├── analyze.py                           # diagnostic EDA
│   └── explain.py                           # SHAP-based XAI plots
├── outputs/
│   ├── submissions/
│   │   ├── submission_default.csv           # raw pipeline output
│   │   └── submission_final.csv             # FINAL — upload this to Kaggle
│   ├── logs/                                # train metadata + history
│   ├── shap/                                # SHAP plots + importance CSVs
│   └── cache/                               # feature parquet cache
├── notebooks/
│   └── run_pipeline.ipynb                   # one-click end-to-end notebook
├── run.py                                   # CLI entry point
├── requirements.txt
└── README.md
```

## Setup

```bash
# Python 3.11+ recommended
pip install -r requirements.txt

# Verify data files
ls data/sales.csv data/sample_submission.csv
```

## Usage

End-to-end pipeline:

```bash
python run.py            # default = run all stages: analyze → train → explain
```

Individual stages:

```bash
python run.py analyze    # diagnostic EDA (no training)
python run.py tune 25    # Optuna hyperparameter search (25 trials per booster × target)
python run.py train      # train pipeline → submission_default.csv (and copy → submission_final.csv)
python run.py explain    # SHAP analysis → outputs/shap/
```

Or run the one-click notebook: `notebooks/run_pipeline.ipynb` (Cell → Run All).

The final submission to upload to Kaggle is **`outputs/submissions/submission_final.csv`**.

## Reproducibility

| Component | Seed | Deterministic? |
|---|---|---|
| LightGBM | 42 | yes (with fixed library version) |
| XGBoost | 42 | yes |
| CatBoost | 42 | mostly yes (±1-2K MAE variance possible) |
| Ridge | 42 | yes |

Re-running the full pipeline reproduces `submission_final.csv` to within ±1-2K MAE.

## Key hyperparameters (in `src/config.py`)

| Constant | Value | Purpose |
|---|---|---|
| `WEIGHT_STABLE_YEARS` | (2014, 2018) | sample-weight 1.0 window |
| `WEIGHT_OTHER` | 0.01 | weight for years outside stable window |
| `Q_BOOST` | 2.0 | weight multiplier for matching-quarter rows in specialist |
| `SPECIALIST_BLEND` | 0.60 | specialist contribution in per-day blend |
| `RIDGE_WEIGHT` | 0.20 | Ridge fraction in two-layer ensemble |
| `GBM_WEIGHTS_REV` | (0.40, 0.40, 0.20) | LGB/XGB/CAT weights for Revenue |
| `GBM_WEIGHTS_COG` | (0.55, 0.25, 0.20) | LGB/XGB/CAT weights for COGS |
| `DEFAULT_CR`, `DEFAULT_CC` | 1.26, 1.32 | calibration multipliers |

## Output sanity checks

After training, expected metrics:

| Metric | Value |
|---|---|
| Public leaderboard (verified) | 681,610 |
| Validation MAE Revenue (hold-out 2022-H2, calibrated) | ~440K |
| Validation MAE COGS (hold-out 2022-H2, calibrated) | ~400K |
