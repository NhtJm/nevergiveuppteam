"""Optuna tuning for booster hyperparameters.

Tunes LightGBM, XGBoost, CatBoost separately on (train ≤ 2022-07-04 → val > 2022-07-04).
Optimizes log1p(target) MAE on val (unbiased — val excluded from training).
Writes best_params.json for train.py to consume.
"""
from __future__ import annotations
import json
import time
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import optuna
from sklearn.metrics import mean_absolute_error

import config as C
import features as F

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_sample_weights(years):
    lo, hi = C.WEIGHT_STABLE_YEARS
    w = np.full(len(years), C.WEIGHT_OTHER)
    w[(years >= lo) & (years <= hi)] = 1.0
    return w


def setup_data():
    sales = pd.read_csv(C.DATA / "sales.csv", parse_dates=["Date"])
    sales = sales.sort_values("Date").reset_index(drop=True)
    feat = F.build_features(sales.Date)
    feat["Revenue"] = sales.Revenue.values
    feat["COGS"] = sales.COGS.values
    cols = F.feature_columns(feat)
    X = feat[cols].values.astype(float)
    y_rev = np.log1p(feat.Revenue.values)
    y_cog = np.log1p(feat.COGS.values)
    dates = feat.Date
    years = feat.year.values
    w = make_sample_weights(years)

    cutoff = pd.Timestamp(C.HOLDOUT_SPLIT)
    fit_idx = (dates <= cutoff).values
    val_idx = (dates > cutoff).values
    return X, y_rev, y_cog, w, fit_idx, val_idx


def tune_lgb(X, y, w, fit_idx, val_idx, n_trials=30):
    def objective(trial):
        params = {
            "objective": "regression",
            "metric": "mae",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": 5,
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 5.0),
            "seed": C.SEED,
            "verbosity": -1,
        }
        train_data = lgb.Dataset(X[fit_idx], y[fit_idx], weight=w[fit_idx])
        val_data = lgb.Dataset(X[val_idx], y[val_idx])
        m = lgb.train(params, train_data, num_boost_round=2000,
                     valid_sets=[val_data],
                     callbacks=[lgb.early_stopping(150, verbose=False),
                               lgb.log_evaluation(0)])
        pred = m.predict(X[val_idx])
        return mean_absolute_error(y[val_idx], pred)

    study = optuna.create_study(direction="minimize",
                               sampler=optuna.samplers.TPESampler(seed=C.SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def tune_xgb(X, y, w, fit_idx, val_idx, n_trials=30):
    def objective(trial):
        params = {
            "objective": "reg:absoluteerror",
            "tree_method": "hist",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "random_state": C.SEED,
            "verbosity": 0,
        }
        dtrain = xgb.DMatrix(X[fit_idx], y[fit_idx], weight=w[fit_idx])
        dval = xgb.DMatrix(X[val_idx], y[val_idx])
        m = xgb.train(params, dtrain, num_boost_round=2000,
                     evals=[(dval, "val")], early_stopping_rounds=150,
                     verbose_eval=False)
        pred = m.predict(dval)
        return mean_absolute_error(y[val_idx], pred)

    study = optuna.create_study(direction="minimize",
                               sampler=optuna.samplers.TPESampler(seed=C.SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def tune_cat(X, y, w, fit_idx, val_idx, n_trials=30):
    def objective(trial):
        params = {
            "loss_function": "MAE",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "random_state": C.SEED,
            "verbose": False,
        }
        m = CatBoostRegressor(iterations=2000, early_stopping_rounds=150, **params)
        m.fit(X[fit_idx], y[fit_idx], sample_weight=w[fit_idx],
              eval_set=(X[val_idx], y[val_idx]), verbose=False)
        pred = m.predict(X[val_idx])
        return mean_absolute_error(y[val_idx], pred)

    study = optuna.create_study(direction="minimize",
                               sampler=optuna.samplers.TPESampler(seed=C.SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def main(n_trials=30):
    print("=" * 70)
    print(f"OPTUNA TUNING — {n_trials} trials × 3 boosters × 2 targets = "
          f"{n_trials * 6} fits")
    print("=" * 70)

    X, y_rev, y_cog, w, fit_idx, val_idx = setup_data()
    print(f"\nTrain {fit_idx.sum()} / Val {val_idx.sum()}")

    best = {"REV": {}, "COG": {}}
    t0 = time.time()
    for tgt, y in [("REV", y_rev), ("COG", y_cog)]:
        print(f"\n--- Target: {tgt} ---")
        for name, fn in [("lgb", tune_lgb), ("xgb", tune_xgb), ("cat", tune_cat)]:
            ts = time.time()
            params, val_mae = fn(X, y, w, fit_idx, val_idx, n_trials=n_trials)
            print(f"  {name}: best val_MAE_log={val_mae:.4f}  "
                  f"params={ {k: round(v, 4) if isinstance(v, float) else v for k, v in params.items()} }  "
                  f"({time.time()-ts:.0f}s)")
            best[tgt][name] = {"params": params, "val_mae_log": val_mae}

    elapsed = time.time() - t0
    print(f"\nTotal tuning time: {elapsed:.0f}s")

    out_path = C.LOGS / "best_params.json"
    with open(out_path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    main(n_trials=n)
