"""Ensemble blending and calibration logic."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

import config as C


def gbm_blend(p_lgb, p_xgb, p_cat, weights, p_lgbq=None):
    """Weighted blend of 3-4 boosters.
    weights = (w_lgb, w_xgb, w_cat) or (w_lgb, w_xgb, w_cat, w_lgbq).
    """
    if len(weights) == 3:
        w_lgb, w_xgb, w_cat = weights
        return w_lgb * p_lgb + w_xgb * p_xgb + w_cat * p_cat
    w_lgb, w_xgb, w_cat, w_lgbq = weights
    if p_lgbq is None:
        # If no Q-50 provided but weight given, redistribute to LGB
        return (w_lgb + w_lgbq) * p_lgb + w_xgb * p_xgb + w_cat * p_cat
    return w_lgb * p_lgb + w_xgb * p_xgb + w_cat * p_cat + w_lgbq * p_lgbq


def specialist_blend(p_specialist, p_base, alpha=C.SPECIALIST_BLEND):
    """Blend Q-specialist with base: alpha × specialist + (1-alpha) × base."""
    return alpha * p_specialist + (1 - alpha) * p_base


def two_layer_ensemble(p_ridge, p_gbm_blend, ridge_w=C.RIDGE_WEIGHT):
    """ridge_w × Ridge + (1 - ridge_w) × GBM_blend."""
    return ridge_w * p_ridge + (1 - ridge_w) * p_gbm_blend


def auto_tune_calibration(raw_preds, y_true, search_range=(0.80, 1.50, 0.01)):
    """Grid-search the optimal scalar multiplier minimizing MAE.

    Args:
        raw_preds: predictions before calibration (in original scale, not log)
        y_true: actual values
        search_range: (start, end, step)

    Returns:
        (best_multiplier, best_mae)
    """
    start, end, step = search_range
    best_mult, best_mae = 1.0, float("inf")
    for mult in np.arange(start, end + step, step):
        mae = mean_absolute_error(y_true, raw_preds * mult)
        if mae < best_mae:
            best_mae = mae
            best_mult = float(mult)
    return best_mult, float(best_mae)


def per_quarter_calibration(raw_preds, y_true, quarters,
                             search_range=(0.80, 1.50, 0.01)):
    """Tune separate calibration per quarter — captures seasonal scale shifts."""
    mults = {}
    for q in (1, 2, 3, 4):
        mask = quarters == q
        if mask.sum() == 0:
            mults[q] = 1.0
            continue
        m, _ = auto_tune_calibration(raw_preds[mask], y_true[mask], search_range)
        mults[q] = m
    return mults


def apply_per_quarter_calibration(preds, quarters, mults):
    out = preds.copy()
    for q in (1, 2, 3, 4):
        out[quarters == q] = preds[quarters == q] * mults[q]
    return out
