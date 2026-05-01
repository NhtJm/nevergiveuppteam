"""Model trainers: LightGBM + XGBoost + CatBoost + Ridge.

Each booster supports:
  - Sample weighted training (focus 2014-2018)
  - Hold-out early stopping then refit on full data
  - Quarterly specialists (boost weight ×2 for target quarter)
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge

import config as C

warnings.filterwarnings("ignore")


# ─── Default hyperparameters (tuned via Optuna pre-runs) ──────────────────
LGB_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.016,
    "num_leaves": 127,
    "min_data_in_leaf": 30,
    "feature_fraction": 0.75,
    "bagging_fraction": 0.85,
    "bagging_freq": 5,
    "lambda_l2": 1.0,
    "seed": C.SEED,
    "verbosity": -1,
}

# LightGBM with quantile-0.5 objective — directly minimizes MAE (Kaggle metric)
LGB_QUANTILE_PARAMS = {
    "objective": "quantile",
    "alpha": 0.5,
    "metric": "mae",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_data_in_leaf": 30,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 5,
    "lambda_l2": 1.0,
    "seed": C.SEED,
    "verbosity": -1,
}


def train_lgb_quantile(X, y, w, dates, custom=None, num_rounds=5000, es=300,
                      return_val_pred=False):
    """LightGBM with quantile-0.5 (median) objective. Direct MAE optimization."""
    params = LGB_QUANTILE_PARAMS.copy()
    if custom:
        params.update(custom)
    fit_idx, val_idx = _split_indices(dates)
    if val_idx.sum() == 0:
        full_data = lgb.Dataset(X, y, weight=w)
        m = lgb.train(params, full_data, num_boost_round=num_rounds)
        if return_val_pred:
            return m, np.zeros(0)
        return m
    train_data = lgb.Dataset(X[fit_idx], y[fit_idx], weight=w[fit_idx])
    val_data = lgb.Dataset(X[val_idx], y[val_idx])
    cb = [lgb.early_stopping(es, verbose=False), lgb.log_evaluation(0)]
    booster_es = lgb.train(
        params, train_data, num_boost_round=num_rounds,
        valid_sets=[val_data], callbacks=cb,
    )
    val_pred = booster_es.predict(X[val_idx]) if return_val_pred else None
    full_data = lgb.Dataset(X, y, weight=w)
    full_model = lgb.train(params, full_data, num_boost_round=booster_es.best_iteration)
    if return_val_pred:
        return full_model, val_pred
    return full_model

XGB_PARAMS = {
    "objective": "reg:absoluteerror",
    "tree_method": "hist",
    "learning_rate": 0.03,
    "max_depth": 6,
    "min_child_weight": 30,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 1.0,
    "random_state": C.SEED,
    "verbosity": 0,
}

CAT_PARAMS = {
    "loss_function": "MAE",
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "random_state": C.SEED,
    "verbose": False,
}


def _split_indices(dates: pd.Series):
    """Train + val mask split at HOLDOUT_SPLIT for early stopping."""
    cutoff = pd.Timestamp(C.HOLDOUT_SPLIT)
    fit_idx = (dates <= cutoff).values
    val_idx = (dates > cutoff).values
    return fit_idx, val_idx


# ─── LightGBM ─────────────────────────────────────────────────────────────
def train_lgb(X, y, w, dates, custom=None, num_rounds=5000, es=300,
              return_val_pred=False):
    """Train LightGBM with hold-out early stopping + refit on full data.

    If return_val_pred=True, returns (full_model, train_only_val_pred).
    The val_pred is from the TRAIN-ONLY model (unbiased — val data
    excluded from training), suitable for calibration tuning.
    """
    params = LGB_PARAMS.copy()
    if custom:
        params.update(custom)
    fit_idx, val_idx = _split_indices(dates)
    if val_idx.sum() == 0:
        full_data = lgb.Dataset(X, y, weight=w)
        m = lgb.train(params, full_data, num_boost_round=num_rounds)
        if return_val_pred:
            return m, np.zeros(0)
        return m
    train_data = lgb.Dataset(X[fit_idx], y[fit_idx], weight=w[fit_idx])
    val_data = lgb.Dataset(X[val_idx], y[val_idx])
    cb = [lgb.early_stopping(es, verbose=False), lgb.log_evaluation(0)]
    booster_es = lgb.train(
        params, train_data, num_boost_round=num_rounds,
        valid_sets=[val_data], callbacks=cb,
    )
    val_pred = booster_es.predict(X[val_idx]) if return_val_pred else None
    full_data = lgb.Dataset(X, y, weight=w)
    full_model = lgb.train(params, full_data, num_boost_round=booster_es.best_iteration)
    if return_val_pred:
        return full_model, val_pred
    return full_model


# ─── XGBoost ──────────────────────────────────────────────────────────────
def train_xgb(X, y, w, dates, custom=None, num_rounds=5000, es=300,
              return_val_pred=False):
    params = XGB_PARAMS.copy()
    if custom:
        params.update(custom)
    fit_idx, val_idx = _split_indices(dates)
    dtrain_full = xgb.DMatrix(X, y, weight=w)
    if val_idx.sum() == 0:
        m = xgb.train(params, dtrain_full, num_boost_round=num_rounds)
        if return_val_pred:
            return m, np.zeros(0)
        return m
    dtrain = xgb.DMatrix(X[fit_idx], y[fit_idx], weight=w[fit_idx])
    dval = xgb.DMatrix(X[val_idx], y[val_idx])
    booster_es = xgb.train(
        params, dtrain, num_boost_round=num_rounds,
        evals=[(dval, "val")], early_stopping_rounds=es, verbose_eval=False,
    )
    val_pred = booster_es.predict(dval) if return_val_pred else None
    full_model = xgb.train(params, dtrain_full,
                           num_boost_round=booster_es.best_iteration)
    if return_val_pred:
        return full_model, val_pred
    return full_model


def xgb_predict(model, X):
    return model.predict(xgb.DMatrix(X))


# ─── CatBoost ─────────────────────────────────────────────────────────────
def train_cat(X, y, w, dates, custom=None, num_rounds=5000, es=300,
              return_val_pred=False):
    params = CAT_PARAMS.copy()
    if custom:
        params.update(custom)
    fit_idx, val_idx = _split_indices(dates)
    if val_idx.sum() == 0:
        m = CatBoostRegressor(iterations=num_rounds, **params)
        m.fit(X, y, sample_weight=w, verbose=False)
        if return_val_pred:
            return m, np.zeros(0)
        return m
    m_es = CatBoostRegressor(iterations=num_rounds, early_stopping_rounds=es, **params)
    m_es.fit(
        X[fit_idx], y[fit_idx], sample_weight=w[fit_idx],
        eval_set=(X[val_idx], y[val_idx]), verbose=False,
    )
    best_iter = m_es.get_best_iteration() or num_rounds
    val_pred = m_es.predict(X[val_idx]) if return_val_pred else None
    m_full = CatBoostRegressor(iterations=best_iter, **params)
    m_full.fit(X, y, sample_weight=w, verbose=False)
    if return_val_pred:
        return m_full, val_pred
    return m_full


# ─── Quarterly Specialist Wrapper ─────────────────────────────────────────
def boost_weights_for_quarter(w_base, quarters, target_q, boost=C.Q_BOOST):
    w = w_base.copy()
    mask = quarters == target_q
    w[mask] = w[mask] * boost
    return w


def train_q_specialist(trainer, X, y, w_base, quarters, dates, target_q, **kw):
    w_q = boost_weights_for_quarter(w_base, quarters, target_q)
    return trainer(X, y, w_q, dates, **kw)


# ─── Ridge with z-score normalization ─────────────────────────────────────
def train_ridge(X_train, y_train, alpha=3.0):
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma == 0] = 1.0
    Xs = (X_train - mu) / sigma
    model = Ridge(alpha=alpha, random_state=C.SEED)
    model.fit(Xs, y_train)
    return model, (mu, sigma)


def predict_ridge(model, X_test, stats):
    mu, sigma = stats
    return model.predict((X_test - mu) / sigma)


# ─── Multi-seed averaging ─────────────────────────────────────────────────
def predict_avg_seeds(trainer, predict_fn, X_full, y, w, dates, X_test,
                     seeds=C.SEEDS, **trainer_kw):
    preds = []
    for s in seeds:
        custom = trainer_kw.get("custom", {}).copy() if trainer_kw.get("custom") else {}
        custom["seed" if trainer is train_lgb else "random_state"] = s
        kw = {**trainer_kw, "custom": custom}
        m = trainer(X_full, y, w, dates, **kw)
        preds.append(predict_fn(m, X_test))
    return np.mean(preds, axis=0)
