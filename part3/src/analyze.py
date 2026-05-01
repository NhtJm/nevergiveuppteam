"""Diagnostic analysis on sales.csv — print key statistics for sanity-checking."""
from __future__ import annotations
import numpy as np
import pandas as pd

import config as C


def main():
    print("=" * 70)
    print("DIAGNOSTIC ANALYSIS — DATA VERIFICATION")
    print("=" * 70)

    sales = pd.read_csv(C.DATA / "sales.csv", parse_dates=["Date"])
    sales = sales.sort_values("Date").reset_index(drop=True)
    sample = pd.read_csv(C.DATA / "sample_submission.csv", parse_dates=["Date"])
    promos = pd.read_csv(C.DATA / "promotions.csv",
                         parse_dates=["start_date", "end_date"])

    print(f"\nTrain: {len(sales)} rows  ({sales.Date.min().date()} → {sales.Date.max().date()})")
    print(f"Test:  {len(sample)} rows  ({sample.Date.min().date()} → {sample.Date.max().date()})")
    print(f"Promotions: {len(promos)} entries")

    print("\n--- Yearly Revenue stats ---")
    yearly = sales.groupby(sales.Date.dt.year).agg(
        days=("Revenue", "count"),
        rev_mean=("Revenue", "mean"),
        rev_total=("Revenue", "sum"),
        cogs_mean=("COGS", "mean"),
    ).round(0)
    yearly["margin_pct"] = (
        (yearly.rev_mean - yearly.cogs_mean) / yearly.rev_mean * 100
    ).round(2)
    yearly["yoy_growth_pct"] = (yearly.rev_total.pct_change() * 100).round(1)
    print(yearly.to_string())

    print("\n--- Day-of-week pattern ---")
    dow = sales.groupby(sales.Date.dt.dayofweek).Revenue.mean().round(0)
    dow.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    print(dow.to_string())

    print("\n--- Monthly Revenue means (across all years) ---")
    monthly = sales.groupby(sales.Date.dt.month).Revenue.mean().round(0)
    print(monthly.to_string())

    print("\n--- COGS / Revenue ratio distribution ---")
    sales["margin"] = sales.COGS / sales.Revenue.clip(lower=1)
    print(sales.margin.describe().round(3).to_string())
    print(f"  days where COGS > Revenue: {(sales.margin > 1).sum()}  "
          f"({(sales.margin > 1).mean() * 100:.1f}%)")

    print("\n--- Promotion months distribution ---")
    pm = promos.start_date.dt.month.value_counts().sort_index()
    print(pm.to_string())

    print("\n--- Outliers (top 10 highest revenue days) ---")
    top = sales.nlargest(10, "Revenue")[["Date", "Revenue", "COGS"]]
    print(top.to_string(index=False))

    print("\n--- Sample weight distribution (training emphasis 2014-2018) ---")
    years = sales.Date.dt.year.values
    ws_yr_lo, ws_yr_hi = C.WEIGHT_STABLE_YEARS
    n_stable = ((years >= ws_yr_lo) & (years <= ws_yr_hi)).sum()
    print(f"  Stable years ({ws_yr_lo}-{ws_yr_hi}): {n_stable} days at weight 1.0")
    print(f"  Other years: {len(years) - n_stable} days at weight {C.WEIGHT_OTHER}")
    eff_w_share = n_stable / (n_stable + (len(years) - n_stable) * C.WEIGHT_OTHER)
    print(f"  Effective weight share to stable: {eff_w_share:.2%}")

    print("\n--- Tet date verification ---")
    for y, t in C.TET_DATES.items():
        in_train = pd.Timestamp(t) <= sales.Date.max()
        marker = "TRAIN" if in_train else "TEST "
        print(f"  {marker}  Tet {y}: {t}")


if __name__ == "__main__":
    main()
