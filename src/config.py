"""Project configuration: paths, constants, sample weights, calibration."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
SUBS = OUT / "submissions"
LOGS = OUT / "logs"
SHAP_DIR = OUT / "shap"
CACHE = OUT / "cache"

for d in (OUT, SUBS, LOGS, SHAP_DIR, CACHE):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
TARGETS = ["Revenue", "COGS"]

# Train/test boundaries from competition
TRAIN_START = "2012-07-04"
TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END = "2024-07-01"

# Hold-out split for early stopping + calibration tuning
HOLDOUT_SPLIT = "2022-07-04"

# Sample weighting: focus on stable mid-period to avoid 2013 noise + COVID-era distortion
WEIGHT_STABLE_YEARS = (2014, 2018)  # weight 1.0
WEIGHT_OTHER = 0.01  # other years

# Default calibration multipliers (tuned per-target on hold-out)
DEFAULT_CR = 1.26  # Revenue
DEFAULT_CC = 1.32  # COGS

# Q-specialist boost factor
Q_BOOST = 2.0
SPECIALIST_BLEND = 0.60  # 60% specialist + 40% base

# Ensemble weights
RIDGE_WEIGHT = 0.20  # Ridge vs GBM_blend
# GBM_blend per target: weights for [LGB, XGB, CAT, LGB-Quantile-50]
# v8: added Quantile-50 as 4th booster — directly optimizes MAE (Kaggle metric)
GBM_WEIGHTS_REV = (0.30, 0.30, 0.15, 0.25)
GBM_WEIGHTS_COG = (0.45, 0.20, 0.15, 0.20)

# Multi-seed ensemble (single seed proven best for spike-heavy seasonal data)
SEEDS = [42]

# VN holidays (fixed dates) — based on EDA of promotions.csv + cultural events
VN_FIXED_HOLIDAYS = [
    (1, 1, "new_year"),
    (3, 8, "womens_day"),
    (4, 30, "reunification"),
    (5, 1, "labor_day"),
    (9, 2, "national_day"),
    (10, 20, "vn_womens_day"),
    (11, 11, "dd_1111"),
    (12, 12, "dd_1212"),
    (12, 24, "christmas_eve"),
    (12, 25, "christmas"),
]

# Lunar New Year (Tet) dates — manually compiled from Vietnamese calendar
TET_DATES = {
    2012: "2012-01-23", 2013: "2013-02-10", 2014: "2014-01-31", 2015: "2015-02-19",
    2016: "2016-02-08", 2017: "2017-01-28", 2018: "2018-02-16", 2019: "2019-02-05",
    2020: "2020-01-25", 2021: "2021-02-12", 2022: "2022-02-01", 2023: "2023-01-22",
    2024: "2024-02-10",
}

# Recurring promotion schedule extracted from promotions.csv
# Format: (name, start_month, start_day, duration_days, discount_pct, recur_rule)
PROMO_SCHEDULE = [
    ("spring_sale", 3, 18, 30, 12, "yearly"),
    ("mid_year", 6, 23, 29, 18, "yearly"),
    ("fall_launch", 8, 30, 32, 10, "yearly"),
    ("year_end", 11, 18, 45, 20, "yearly"),
    ("urban_blowout", 7, 30, 33, 50, "odd"),  # only odd years
    ("rural_special", 1, 30, 30, 15, "odd"),
]
