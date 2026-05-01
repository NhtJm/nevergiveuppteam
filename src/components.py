"""Component decomposition pipeline.

Idea: Revenue = #orders/day × avg_order_value/day.
Train two separate models — one for daily order count, one for AOV — then
multiply for a Revenue forecast that's an alternative to direct prediction.

The two underlying quantities have very different distributions:
  - n_orders: integer count, high variance, ranges 0-750
  - AOV: continuous, stable ~25K, low variance

Predicting them separately tends to be more robust than predicting Revenue
directly, especially over long horizons.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

import config as C
import features as F
import models as M


def build_daily_components(force=False):
    """Compute daily n_orders and AOV from raw transaction data.
    Cached as parquet to avoid recomputation.
    """
    cache = C.CACHE / "daily_components.parquet"
    if cache.exists() and not force:
        return pd.read_parquet(cache)

    oi = pd.read_csv(C.DATA / "order_items.csv", low_memory=False)
    orders = pd.read_csv(C.DATA / "orders.csv", parse_dates=["order_date"])
    completed = orders[orders.order_status != "cancelled"].copy()

    oi_dated = oi.merge(
        completed[["order_id", "order_date"]], on="order_id", how="inner"
    )
    oi_dated["line_revenue"] = (
        oi_dated.quantity * oi_dated.unit_price
        - oi_dated.discount_amount.fillna(0)
    )

    daily_orders = (
        completed.groupby("order_date")
        .order_id.nunique()
        .reset_index()
        .rename(columns={"order_date": "Date", "order_id": "n_orders"})
    )
    daily_rev = (
        oi_dated.groupby("order_date")
        .line_revenue.sum()
        .reset_index()
        .rename(columns={"order_date": "Date", "line_revenue": "rev_recon"})
    )

    df = daily_orders.merge(daily_rev, on="Date")
    df["aov"] = df.rev_recon / df.n_orders.clip(lower=1)
    df.to_parquet(cache)
    return df


def make_sample_weights(years):
    lo, hi = C.WEIGHT_STABLE_YEARS
    w = np.full(len(years), C.WEIGHT_OTHER)
    w[(years >= lo) & (years <= hi)] = 1.0
    return w


def train_component_models(target_name, train_df, test_df, feat_cols,
                            seeds=(42, 123, 777, 2026, 314)):
    """Train multi-seed LGB+XGB+CAT ensemble for one component (n_orders or AOV).
    Returns averaged test predictions.
    """
    X_tr = train_df[feat_cols].fillna(0).values.astype(float)
    y_tr = np.log1p(train_df[target_name].values)
    X_te = test_df[feat_cols].fillna(0).values.astype(float)
    dates_tr = train_df.Date
    years_tr = train_df.Date.dt.year.values
    w_full = make_sample_weights(years_tr)

    test_preds = []
    for seed in seeds:
        # LGB
        m_lgb = M.train_lgb(X_tr, y_tr, w_full, dates_tr,
                            custom={"seed": seed})
        p_lgb = np.expm1(m_lgb.predict(X_te))
        # XGB
        m_xgb = M.train_xgb(X_tr, y_tr, w_full, dates_tr,
                            custom={"random_state": seed})
        p_xgb = np.expm1(M.xgb_predict(m_xgb, X_te))
        # CAT
        m_cat = M.train_cat(X_tr, y_tr, w_full, dates_tr,
                            custom={"random_state": seed})
        p_cat = np.expm1(m_cat.predict(X_te))
        # Equal blend per seed
        test_preds.append((p_lgb + p_xgb + p_cat) / 3)

    # Average across seeds
    avg = np.mean(test_preds, axis=0)
    return np.maximum(avg, 0)


def main():
    """Entry point — produces submission_components.csv."""
    import time
    t0 = time.time()
    print("=" * 70)
    print("COMPONENT DECOMPOSITION PIPELINE — Revenue = orders × AOV")
    print("=" * 70)

    # Load static features for both train and test
    sales = pd.read_csv(C.DATA / "sales.csv", parse_dates=["Date"])
    sales = sales.sort_values("Date").reset_index(drop=True)
    test_dates = pd.date_range(C.TEST_START, C.TEST_END, freq="D")

    print(f"\n[1/4] Building static features...")
    feat_train = F.build_features(sales.Date)
    feat_test = F.build_features(test_dates)

    # Merge component data into train features
    components = build_daily_components()
    feat_train = feat_train.merge(
        components[["Date", "n_orders", "aov"]], on="Date", how="left"
    )
    # Fill any missing days
    feat_train["n_orders"] = feat_train.n_orders.fillna(0)
    feat_train["aov"] = feat_train.aov.fillna(feat_train.aov.median())

    cols = F.feature_columns(feat_train)
    cols = [c for c in cols if c not in ("n_orders", "aov", "rev_recon")]
    print(f"  features: {len(cols)}  train rows: {len(feat_train)}")

    # ── Train 2 component models with multi-seed ──────────────────────────
    print(f"\n[2/4] Training n_orders model (5 seeds × 3 boosters = 15 fits)...")
    n_orders_pred = train_component_models("n_orders", feat_train, feat_test, cols)
    print(f"  n_orders predicted mean: {n_orders_pred.mean():.1f}")

    print(f"\n[3/4] Training AOV model (5 seeds × 3 boosters = 15 fits)...")
    aov_pred = train_component_models("aov", feat_train, feat_test, cols)
    print(f"  AOV predicted mean: {aov_pred.mean():,.0f}")

    # ── Combine: Revenue = orders × AOV ───────────────────────────────────
    print(f"\n[4/4] Combining Revenue = orders × AOV...")
    rev_component = n_orders_pred * aov_pred
    print(f"  Component Revenue mean: {rev_component.mean():,.0f}")

    # Apply target-mean anchor scaling for robustness across long horizon
    TARGET_REV_MEAN = 4_400_000.0
    scale_factor = TARGET_REV_MEAN / rev_component.mean()
    print(f"  Scale to TARGET_MEAN={TARGET_REV_MEAN:,.0f}  factor={scale_factor:.3f}")
    rev_component_scaled = rev_component * scale_factor

    # COGS via train-period margin ratio
    margin_ratio = (sales.COGS / sales.Revenue.clip(lower=1)).median()
    cogs_component = rev_component_scaled * margin_ratio
    cogs_component = np.minimum(cogs_component, rev_component_scaled * 0.98)

    sub = pd.DataFrame({
        "Date": test_dates.strftime("%Y-%m-%d"),
        "Revenue": np.round(rev_component_scaled, 2),
        "COGS": np.round(cogs_component, 2),
    })
    out_path = C.SUBS / "submission_components.csv"
    sub.to_csv(out_path, index=False)

    elapsed = time.time() - t0
    print(f"\n[DONE] {elapsed:.1f}s elapsed")
    print(f"  Wrote {out_path}")
    print(f"  Rev mean: {sub.Revenue.mean():,.0f}  COGS mean: {sub.COGS.mean():,.0f}")
    print(f"  margin: {1 - (sub.COGS / sub.Revenue.clip(1)).mean():.3f}")


if __name__ == "__main__":
    main()
