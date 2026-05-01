"""Training pipeline (v4): unbiased calibration tuning + per-quarter calibration.

Architecture:
  Layer 1: 2 GBM boosters (LGB + XGB), each with base + 4 quarterly specialists.
           Single seed. Each model returns (full_model, train_only_val_pred).
  Layer 2: Ridge on z-score normalized features.
  Layer 3: Two-layer ensemble (Ridge + GBM_blend) per target.
  Layer 4: PROPER auto-tuned calibration:
           - Global CR/CC tuned on UNBIASED val predictions (train-only model)
           - Per-quarter CR/CC for finer adjustment
           - Picks best variant by val MAE

Sample weighting: 2014-2018 → 1.0, others → 0.01.
Target: log1p.
Features: ~88 calendar+Fourier+Tet+per-promo (NO Tet-aligned lag — proven to hurt this architecture).
"""
from __future__ import annotations
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

import config as C
import features as F
import models as M
from ensemble import (
    gbm_blend, specialist_blend, two_layer_ensemble,
    auto_tune_calibration, per_quarter_calibration,
    apply_per_quarter_calibration,
)


def make_sample_weights(years):
    lo, hi = C.WEIGHT_STABLE_YEARS
    w = np.full(len(years), C.WEIGHT_OTHER)
    w[(years >= lo) & (years <= hi)] = 1.0
    return w


def train_with_unbiased_val(trainer, predict_fn, X_full, y, w, dates,
                            X_test, X_val_features):
    """Train base model, return (test_pred_log, val_pred_log_unbiased)."""
    full_model, val_pred_log = trainer(X_full, y, w, dates, return_val_pred=True)
    test_pred_log = predict_fn(full_model, X_test)
    return test_pred_log, val_pred_log


def train_specialists_with_val(trainer, predict_fn, X_full, y, w_base, dates,
                               quarters_tr, X_test, quarters_te,
                               X_val_features, quarters_val, val_idx,
                               custom=None):
    """4 quarterly specialists. Returns:
       test_pred_blended (specialist-quarter-matched test predictions)
       val_pred_blended (specialist-quarter-matched val predictions, train-only)
    """
    test_pred = np.zeros(len(X_test))
    val_pred = np.zeros(val_idx.sum())

    for q in (1, 2, 3, 4):
        w_q = M.boost_weights_for_quarter(w_base, quarters_tr, q)
        full_m, val_pred_q_log = trainer(
            X_full, y, w_q, dates, return_val_pred=True,
            num_rounds=3000, es=200, custom=custom,
        )
        test_pred_q = np.expm1(predict_fn(full_m, X_test))
        val_pred_q = np.expm1(val_pred_q_log)
        # Mask to specialist quarter
        test_pred[quarters_te == q] = test_pred_q[quarters_te == q]
        val_pred[quarters_val == q] = val_pred_q[quarters_val == q]

    return test_pred, val_pred


def main():
    t0 = time.time()
    print("=" * 70)
    print("TRAINING PIPELINE v4 — unbiased calibration + per-quarter tuning")
    print("=" * 70)

    # ── Load + features ───────────────────────────────────────────────────
    sales = pd.read_csv(C.DATA / "sales.csv", parse_dates=["Date"])
    sales = sales.sort_values("Date").reset_index(drop=True)
    test_dates = pd.date_range(C.TEST_START, C.TEST_END, freq="D")

    print(f"\n[1/5] Building features (NO Tet-aligned lag — hurt in this arch)...")
    feat_train = F.build_features(sales.Date)  # no train_lookup → skip Tet lag
    feat_train["Revenue"] = sales.Revenue.values
    feat_train["COGS"] = sales.COGS.values
    feat_test = F.build_features(test_dates)
    cols = F.feature_columns(feat_train)

    # XAI-driven feature selection: keep top-N by combined SHAP importance
    shap_imp_path = C.SHAP_DIR / "shap_importance_combined.csv"
    top_n = int(__import__("os").environ.get("TOP_N_FEATURES", "0"))
    if top_n > 0 and shap_imp_path.exists():
        imp = pd.read_csv(shap_imp_path).sort_values("total", ascending=False)
        keep = set(imp.head(top_n).feature.tolist())
        cols_kept = [c for c in cols if c in keep]
        dropped = sorted(set(cols) - keep)
        print(f"      [XAI] Keeping top-{top_n} features: {len(cols_kept)}/{len(cols)}")
        print(f"      [XAI] Dropped: {len(dropped)} features (e.g., {dropped[:5]}...)")
        cols = cols_kept

    print(f"      train: {len(feat_train)} rows × {len(cols)} features")

    X_tr = feat_train[cols].values.astype(float)
    X_te = feat_test[cols].values.astype(float)
    y_rev = np.log1p(feat_train.Revenue.values)
    y_cog = np.log1p(feat_train.COGS.values)
    dates_tr = feat_train.Date
    years_tr = feat_train.year.values
    quarters_tr = feat_train.quarter.values
    quarters_te = feat_test.quarter.values

    w_full = make_sample_weights(years_tr)

    # Hold-out indices
    val_mask = (dates_tr > pd.Timestamp(C.HOLDOUT_SPLIT)) & \
               (dates_tr <= pd.Timestamp(C.TRAIN_END))
    val_idx = val_mask.values
    X_val = X_tr[val_idx]
    quarters_val = quarters_tr[val_idx]
    y_val_rev = feat_train.Revenue.values[val_idx]
    y_val_cog = feat_train.COGS.values[val_idx]
    print(f"      val period (2022-H2): {val_idx.sum()} days")

    # ── Layer 1 — boosters ────────────────────────────────────────────────
    print(f"\n[2/5] Layer 1 — LGB + XGB × (base + 4 Q-specialists), unbiased val pred")

    p_test = {"REV": {}, "COG": {}}
    p_val = {"REV": {}, "COG": {}}

    # Load tuned hyperparameters if available
    tuned_path = C.LOGS / "best_params.json"
    tuned_params = None
    if tuned_path.exists():
        with open(tuned_path) as f:
            tuned_params = json.load(f)
        print(f"      Loaded tuned hyperparameters from {tuned_path}")

    boosters = [
        ("LGB", M.train_lgb, lambda m, X: m.predict(X), 0),
        ("XGB", M.train_xgb, M.xgb_predict, 1),
        ("CAT", M.train_cat, lambda m, X: m.predict(X), 2),
        ("LGBQ", M.train_lgb_quantile, lambda m, X: m.predict(X), 3),
    ]
    targets = [("REV", y_rev, C.GBM_WEIGHTS_REV),
               ("COG", y_cog, C.GBM_WEIGHTS_COG)]

    for tgt, y, weights in targets:
        for name, trainer, predict_fn, idx in boosters:
            if weights[idx] == 0:
                p_test[tgt][name] = np.zeros(len(X_te))
                p_val[tgt][name] = np.zeros(val_idx.sum())
                continue

            # Inject tuned hyperparams if available for this (target, model)
            custom = None
            if tuned_params is not None:
                key = name.lower()
                if tgt in tuned_params and key in tuned_params[tgt]:
                    custom = tuned_params[tgt][key].get("params", {}).copy()

            print(f"  {name}-{tgt}: base..." + ("  [TUNED]" if custom else ""))
            full_m, val_log = trainer(X_tr, y, w_full, dates_tr,
                                     return_val_pred=True, custom=custom)
            p_base_test = np.expm1(predict_fn(full_m, X_te))
            p_base_val = np.expm1(val_log)

            print(f"  {name}-{tgt}: 4 Q-specialists...")
            p_spec_test, p_spec_val = train_specialists_with_val(
                trainer, predict_fn, X_tr, y, w_full, dates_tr,
                quarters_tr, X_te, quarters_te, X_val, quarters_val, val_idx,
                custom=custom,
            )

            p_test[tgt][name] = specialist_blend(p_spec_test, p_base_test)
            p_val[tgt][name] = specialist_blend(p_spec_val, p_base_val)

    # GBM blend (4 boosters: LGB + XGB + CAT + LGB-Quantile-50)
    p_gbm_rev_test = gbm_blend(
        p_test["REV"]["LGB"], p_test["REV"]["XGB"], p_test["REV"]["CAT"],
        C.GBM_WEIGHTS_REV, p_lgbq=p_test["REV"].get("LGBQ"),
    )
    p_gbm_cog_test = gbm_blend(
        p_test["COG"]["LGB"], p_test["COG"]["XGB"], p_test["COG"]["CAT"],
        C.GBM_WEIGHTS_COG, p_lgbq=p_test["COG"].get("LGBQ"),
    )
    p_gbm_rev_val = gbm_blend(
        p_val["REV"]["LGB"], p_val["REV"]["XGB"], p_val["REV"]["CAT"],
        C.GBM_WEIGHTS_REV, p_lgbq=p_val["REV"].get("LGBQ"),
    )
    p_gbm_cog_val = gbm_blend(
        p_val["COG"]["LGB"], p_val["COG"]["XGB"], p_val["COG"]["CAT"],
        C.GBM_WEIGHTS_COG, p_lgbq=p_val["COG"].get("LGBQ"),
    )

    # ── Layer 2 — Ridge ───────────────────────────────────────────────────
    print(f"\n[3/5] Layer 2 — Ridge (z-score)...")
    # Train Ridge on train-only data for unbiased val pred
    fit_mask = (dates_tr <= pd.Timestamp(C.HOLDOUT_SPLIT)).values
    ridge_rev_es, stats_rev_es = M.train_ridge(X_tr[fit_mask], y_rev[fit_mask])
    ridge_cog_es, stats_cog_es = M.train_ridge(X_tr[fit_mask], y_cog[fit_mask])
    p_rd_rev_val = np.expm1(M.predict_ridge(ridge_rev_es, X_val, stats_rev_es))
    p_rd_cog_val = np.expm1(M.predict_ridge(ridge_cog_es, X_val, stats_cog_es))
    # Refit Ridge on full data for test
    ridge_rev_full, stats_rev_full = M.train_ridge(X_tr, y_rev)
    ridge_cog_full, stats_cog_full = M.train_ridge(X_tr, y_cog)
    p_rd_rev_test = np.expm1(M.predict_ridge(ridge_rev_full, X_te, stats_rev_full))
    p_rd_cog_test = np.expm1(M.predict_ridge(ridge_cog_full, X_te, stats_cog_full))

    # ── Layer 3 — Ensemble ────────────────────────────────────────────────
    print(f"\n[4/5] Layer 3 — Ensemble (Ridge + GBM_blend)...")
    raw_rev_test = two_layer_ensemble(p_rd_rev_test, p_gbm_rev_test)
    raw_cog_test = two_layer_ensemble(p_rd_cog_test, p_gbm_cog_test)
    raw_rev_val = two_layer_ensemble(p_rd_rev_val, p_gbm_rev_val)
    raw_cog_val = two_layer_ensemble(p_rd_cog_val, p_gbm_cog_val)
    print(f"      raw test Revenue mean: {raw_rev_test.mean():,.0f}")
    print(f"      raw test COGS mean:    {raw_cog_test.mean():,.0f}")
    print(f"      raw val Revenue mean:  {raw_rev_val.mean():,.0f}  truth={y_val_rev.mean():,.0f}")
    print(f"      raw val COGS mean:     {raw_cog_val.mean():,.0f}  truth={y_val_cog.mean():,.0f}")

    # ── Layer 4 — UNBIASED calibration tuning ─────────────────────────────
    print(f"\n[5/5] Layer 4 — Unbiased calibration tuning on hold-out 2022-H2...")
    cr_global, cr_mae = auto_tune_calibration(raw_rev_val, y_val_rev,
                                              search_range=(0.80, 1.60, 0.005))
    cc_global, cc_mae = auto_tune_calibration(raw_cog_val, y_val_cog,
                                              search_range=(0.80, 1.60, 0.005))
    print(f"      Global CR={cr_global:.3f}  val_MAE_rev={cr_mae:,.0f}")
    print(f"      Global CC={cc_global:.3f}  val_MAE_cog={cc_mae:,.0f}")

    # Per-quarter calibration
    cr_perq = per_quarter_calibration(raw_rev_val, y_val_rev, quarters_val,
                                      search_range=(0.80, 1.60, 0.005))
    cc_perq = per_quarter_calibration(raw_cog_val, y_val_cog, quarters_val,
                                      search_range=(0.80, 1.60, 0.005))
    cr_perq_pred = apply_per_quarter_calibration(raw_rev_val, quarters_val, cr_perq)
    cc_perq_pred = apply_per_quarter_calibration(raw_cog_val, quarters_val, cc_perq)
    cr_perq_mae = mean_absolute_error(y_val_rev, cr_perq_pred)
    cc_perq_mae = mean_absolute_error(y_val_cog, cc_perq_pred)
    print(f"      Per-Q CR={ {q: round(cr_perq[q], 3) for q in cr_perq} }  "
          f"val_MAE_rev={cr_perq_mae:,.0f}")
    print(f"      Per-Q CC={ {q: round(cc_perq[q], 3) for q in cc_perq} }  "
          f"val_MAE_cog={cc_perq_mae:,.0f}")

    # Default for comparison
    cr_dflt_mae = mean_absolute_error(y_val_rev, raw_rev_val * C.DEFAULT_CR)
    cc_dflt_mae = mean_absolute_error(y_val_cog, raw_cog_val * C.DEFAULT_CC)
    print(f"      Default CR={C.DEFAULT_CR}  val_MAE_rev={cr_dflt_mae:,.0f}")
    print(f"      Default CC={C.DEFAULT_CC}  val_MAE_cog={cc_dflt_mae:,.0f}")

    # ── Generate multiple submissions ─────────────────────────────────────
    submissions = {
        "global": (raw_rev_test * cr_global, raw_cog_test * cc_global),
        "per_q": (apply_per_quarter_calibration(raw_rev_test, quarters_te, cr_perq),
                  apply_per_quarter_calibration(raw_cog_test, quarters_te, cc_perq)),
        "default": (raw_rev_test * C.DEFAULT_CR, raw_cog_test * C.DEFAULT_CC),
    }

    for label, (rev, cog) in submissions.items():
        sub = pd.DataFrame({
            "Date": test_dates.strftime("%Y-%m-%d"),
            "Revenue": rev.clip(min=0).round(2),
            "COGS": cog.clip(min=0).round(2),
        })
        path = C.SUBS / f"submission_{label}.csv"
        sub.to_csv(path, index=False)
        print(f"  Wrote {path}  Rev mean={sub.Revenue.mean():,.0f}  "
              f"COGS mean={sub.COGS.mean():,.0f}")

    # Pick best by val MAE
    val_maes = {"global": cr_mae + cc_mae, "per_q": cr_perq_mae + cc_perq_mae,
                "default": cr_dflt_mae + cc_dflt_mae}
    best_label = min(val_maes, key=val_maes.get)
    rev, cog = submissions[best_label]
    sub = pd.DataFrame({
        "Date": test_dates.strftime("%Y-%m-%d"),
        "Revenue": rev.clip(min=0).round(2),
        "COGS": cog.clip(min=0).round(2),
    })
    sub.to_csv(C.SUBS / "submission.csv", index=False)
    print(f"\n[DONE] {time.time() - t0:.1f}s elapsed")
    print(f"  Best variant by val MAE: {best_label}  (sum_MAE={val_maes[best_label]:,.0f})")
    print(f"  MAIN: outputs/submissions/submission.csv  Rev={sub.Revenue.mean():,.0f}  "
          f"COGS={sub.COGS.mean():,.0f}")

    meta = {
        "elapsed_sec": time.time() - t0,
        "n_features": len(cols),
        "best_variant": best_label,
        "val_mae_rev": {"global": cr_mae, "per_q": cr_perq_mae, "default": float(cr_dflt_mae)},
        "val_mae_cog": {"global": cc_mae, "per_q": cc_perq_mae, "default": float(cc_dflt_mae)},
        "calibration_global": {"CR": cr_global, "CC": cc_global},
        "calibration_per_q": {"CR": {int(k): v for k, v in cr_perq.items()},
                              "CC": {int(k): v for k, v in cc_perq.items()}},
        "raw_means_test": {"Revenue": float(raw_rev_test.mean()),
                          "COGS": float(raw_cog_test.mean())},
    }
    with open(C.LOGS / "train_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)


if __name__ == "__main__":
    main()
