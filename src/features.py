"""Feature engineering — calendar + regime + Fourier + Tet + per-promo windows.

Total ~85 features built from Date column only (calendar-only approach).
Anti-leakage: no future-looking aggregates, no per-row lags from sales.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import config as C

TAU = 2 * np.pi


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    d = pd.to_datetime(df.Date)
    df["year"] = d.dt.year
    df["month"] = d.dt.month
    df["day"] = d.dt.day
    df["dow"] = d.dt.dayofweek
    df["doy"] = d.dt.dayofyear
    df["quarter"] = d.dt.quarter
    df["dim"] = d.dt.days_in_month
    df["is_weekend"] = (df.dow >= 5).astype(int)

    # Edge-of-month indicators
    df["days_to_eom"] = df.dim - df.day
    df["days_from_som"] = df.day - 1
    for k in (1, 2, 3):
        df[f"is_last{k}"] = (df.days_to_eom <= k - 1).astype(int)
        df[f"is_first{k}"] = (df.days_from_som <= k - 1).astype(int)

    # Payday indicators (VN salary cycles)
    df["is_payday"] = df.day.isin([1, 15]).astype(int)
    df["is_month_end_zone"] = (df.day >= 28).astype(int)

    # Linear trend baseline (reset to 2020 to focus post-COVID era)
    df["t_days"] = (d - pd.Timestamp("2020-01-01")).dt.days
    df["t_years"] = df.t_days / 365.25

    # Regime indicators: pre-2019, 2019, post-2019 (capture structural breaks)
    df["regime_pre2019"] = (df.year <= 2018).astype(int)
    df["regime_2019"] = (df.year == 2019).astype(int)
    df["regime_post2019"] = (df.year >= 2020).astype(int)
    df["is_odd_year"] = (df.year % 2).astype(int)

    return df


def add_fourier(df: pd.DataFrame) -> pd.DataFrame:
    """Fourier seasonal encodings for yearly/weekly/monthly cycles."""
    df = df.copy()
    # Yearly k=1..5 (multi-harmonic to capture sharp Tet/Black-Friday transitions)
    for k in (1, 2, 3, 4, 5):
        df[f"sin_y{k}"] = np.sin(TAU * k * df.doy / 365.25)
        df[f"cos_y{k}"] = np.cos(TAU * k * df.doy / 365.25)
    # Weekly k=1..2
    for k in (1, 2):
        df[f"sin_w{k}"] = np.sin(TAU * k * df.dow / 7.0)
        df[f"cos_w{k}"] = np.cos(TAU * k * df.dow / 7.0)
    # Monthly k=1..2 (within-month cycle)
    for k in (1, 2):
        df[f"sin_m{k}"] = np.sin(TAU * k * (df.day - 1) / df.dim)
        df[f"cos_m{k}"] = np.cos(TAU * k * (df.day - 1) / df.dim)
    return df


def add_holidays(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for (m, dd_, name) in C.VN_FIXED_HOLIDAYS:
        df[f"hol_{name}"] = ((df.month == m) & (df.day == dd_)).astype(int)

    # Black Friday: last Friday of November
    d = pd.to_datetime(df.Date)

    def _is_black_friday(date):
        if date.month != 11:
            return 0
        last = pd.Timestamp(year=date.year, month=11, day=30)
        last_fri = last - pd.Timedelta(days=(last.dayofweek - 4) % 7)
        return int(date == last_fri)

    df["hol_black_friday"] = [_is_black_friday(dd) for dd in d]
    return df


def add_tet(df: pd.DataFrame) -> pd.DataFrame:
    """Lunar New Year features — windows and signed offset."""
    df = df.copy()
    d = pd.to_datetime(df.Date)
    tet_ts = {y: pd.Timestamp(t) for y, t in C.TET_DATES.items()}

    def _nearest_tet_diff(date):
        cands = []
        for yy in (date.year - 1, date.year, date.year + 1):
            if yy in tet_ts:
                cands.append(tet_ts[yy])
        valid = [(date - c).days for c in cands if abs((date - c).days) <= 60]
        return min(valid, key=abs) if valid else 999

    diffs = np.array([_nearest_tet_diff(dd) for dd in d])
    df["tet_days_diff"] = diffs
    df["tet_in_7"] = (np.abs(diffs) <= 7).astype(int)
    df["tet_in_14"] = (np.abs(diffs) <= 14).astype(int)
    df["tet_in_21"] = (np.abs(diffs) <= 21).astype(int)
    df["tet_before_7"] = ((diffs >= -7) & (diffs < 0)).astype(int)
    df["tet_after_7"] = ((diffs > 0) & (diffs <= 7)).astype(int)
    df["tet_on"] = (diffs == 0).astype(int)
    return df


def add_promo_windows(df: pd.DataFrame) -> pd.DataFrame:
    """Per-promo binary flags + days since/until + discount value.
    Synthetic: extends recurring schedule beyond train data."""
    df = df.copy()
    d = pd.to_datetime(df.Date)
    years = sorted(set(df.year.tolist()))

    for (name, sm, sd, dur, disc, recur) in C.PROMO_SCHEDULE:
        in_prom = np.zeros(len(df), dtype=int)
        since = np.full(len(df), -1.0)
        until = np.full(len(df), -1.0)
        discount = np.zeros(len(df))

        for y in range(min(years) - 1, max(years) + 2):
            if recur == "odd" and y % 2 == 0:
                continue
            try:
                start = pd.Timestamp(year=y, month=sm, day=sd)
            except ValueError:
                continue
            end = start + pd.Timedelta(days=dur)
            mask = (d >= start) & (d <= end)
            mask_arr = mask.values if hasattr(mask, "values") else mask
            in_prom[mask_arr] = 1
            since_vals = (d[mask] - start).dt.days.values
            until_vals = (end - d[mask]).dt.days.values
            since[mask_arr] = since_vals
            until[mask_arr] = until_vals
            discount[mask_arr] = disc or 0
        df[f"promo_{name}"] = in_prom
        df[f"promo_{name}_since"] = since
        df[f"promo_{name}_until"] = until
        df[f"promo_{name}_disc"] = discount
    return df


def add_real_promo_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate from actual promotions.csv: count active, total discount, category flags."""
    df = df.copy()
    d = pd.to_datetime(df.Date)
    promos = pd.read_csv(
        C.DATA / "promotions.csv", parse_dates=["start_date", "end_date"]
    )

    n_active = np.zeros(len(df), dtype=int)
    sum_disc = np.zeros(len(df))
    has_streetwear = np.zeros(len(df), dtype=int)
    has_outdoor = np.zeros(len(df), dtype=int)

    for i, day in enumerate(d):
        active = promos[(promos.start_date <= day) & (promos.end_date >= day)]
        if len(active):
            n_active[i] = len(active)
            sum_disc[i] = active.discount_value.sum()
            has_streetwear[i] = int((active.applicable_category == "Streetwear").any())
            has_outdoor[i] = int((active.applicable_category == "Outdoor").any())

    df["n_active_promos"] = n_active
    df["sum_discount_active"] = sum_disc
    df["has_streetwear_promo"] = has_streetwear
    df["has_outdoor_promo"] = has_outdoor
    return df


def add_tet_aligned_lag(df: pd.DataFrame, train_lookup: pd.DataFrame) -> pd.DataFrame:
    """For each row at date D, lookup Revenue/COGS at the most recent prior train
    year's date with the SAME signed offset from its Tet. Captures Tet-specific
    YoY anchoring (e.g., 2024-02-10 = Tet 2024 → lookup Tet 2022 = 2022-02-01)."""
    df = df.copy()
    d = pd.to_datetime(df.Date)
    rev_lookup = dict(zip(pd.to_datetime(train_lookup.Date), train_lookup.Revenue))
    cogs_lookup = dict(zip(pd.to_datetime(train_lookup.Date), train_lookup.COGS))
    tet_ts = {y: pd.Timestamp(t) for y, t in C.TET_DATES.items()}
    train_years = sorted([y for y in tet_ts if y <= 2022])

    rev_aligned, cogs_aligned = [], []
    for date in d:
        y = date.year
        delta = int((date - tet_ts[y]).days) if y in tet_ts else 0
        ref_y = max([yy for yy in train_years if yy < y], default=2022)
        ref_date = tet_ts[ref_y] + pd.Timedelta(days=delta)
        rev_aligned.append(rev_lookup.get(ref_date, np.nan))
        cogs_aligned.append(cogs_lookup.get(ref_date, np.nan))
    df["rev_tet_aligned_lag"] = rev_aligned
    df["cogs_tet_aligned_lag"] = cogs_aligned
    return df


def build_features(dates, train_lookup: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build full feature matrix from Date column.

    Args:
        dates: array-like of dates to compute features for
        train_lookup: train sales DataFrame [Date, Revenue, COGS] used for
                      Tet-aligned lag features. Optional — if None, lag
                      features are skipped.
    """
    df = pd.DataFrame({"Date": pd.to_datetime(dates)})
    df = add_calendar(df)
    df = add_fourier(df)
    df = add_holidays(df)
    df = add_tet(df)
    df = add_promo_windows(df)
    df = add_real_promo_aggregates(df)
    if train_lookup is not None:
        df = add_tet_aligned_lag(df, train_lookup)
    return df


NON_FEATURES = {"Date", "Revenue", "COGS"}


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURES]
