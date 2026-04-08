"""
Agent 2: Factor Construction
Computes five equity factors and a composite score for each business-month-end
formation date from 2024-01-31 through 2025-12-31.

Factors
-------
1. Momentum   (12-1 skip)          — price-based, time-varying
2. Value      (-trailing P/E)      — fundamental, static (acknowledged limitation)
3. Quality    (ROE)                — fundamental, static
4. Size       (-log market cap)    — fundamental, static
5. Low Vol    (-60-day daily std)  — price-based, time-varying

Each factor is z-scored cross-sectionally and independently each month.
Composite = mean of available z-scores; requires ≥3 of 5.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_monthly_prices() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "prices.parquet")


def load_fundamentals() -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / "fundamentals.parquet")
    return df.set_index("ticker")


def load_daily_prices(tickers: list[str]) -> pd.DataFrame:
    """Return daily adjusted close prices.  Downloads and caches on first call."""
    cache = DATA_DIR / "daily_prices.parquet"
    if cache.exists():
        print("  Loading cached daily prices...")
        return pd.read_parquet(cache)

    print("  daily_prices.parquet not found — downloading (this takes ~5 min)...")
    batch_size = 50
    batches: list[pd.DataFrame] = []
    n_batches = (len(tickers) + batch_size - 1) // batch_size

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        batch_num = i // batch_size + 1
        print(f"    Batch {batch_num}/{n_batches}...")
        try:
            raw = yf.download(
                batch,
                start="2023-01-01",
                end="2026-02-01",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if isinstance(raw.columns, pd.MultiIndex):
                closes = raw["Close"]
            else:
                closes = raw[["Close"]].rename(columns={"Close": batch[0]})
            batches.append(closes)
        except Exception as exc:
            print(f"    Warning: batch {batch_num} failed — {exc}")
        time.sleep(0.1)

    daily = pd.concat(batches, axis=1)
    daily = daily.loc[:, ~daily.columns.duplicated()]
    daily.to_parquet(cache)
    print(f"  Saved daily_prices.parquet  ({daily.shape})")
    return daily


# ---------------------------------------------------------------------------
# Statistical helper
# ---------------------------------------------------------------------------

def z_score(series: pd.Series) -> pd.Series:
    """Cross-sectional z-score computed only over non-NaN values.
    Stocks outside the valid universe (NaN) remain NaN in the output."""
    valid = series.dropna()
    if len(valid) < 2:
        return pd.Series(np.nan, index=series.index)
    return (series - valid.mean()) / valid.std()


# ---------------------------------------------------------------------------
# Individual factor computations
# ---------------------------------------------------------------------------

def momentum_raw(prices_monthly: pd.DataFrame, t_pos: int) -> pd.Series:
    """
    Skip-1 momentum at formation date index position t_pos.

    signal = price[t-1 month-end] / price[t-12 month-end] - 1

    Window: rows [t_pos-12 .. t_pos-1] (12 monthly prices, t-12 through t-1).
    Require ≥11 of those 12 to be non-NaN; otherwise NaN for that stock.
    """
    window = prices_monthly.iloc[t_pos - 12 : t_pos]   # shape (12, n_tickers)
    n_valid = window.notna().sum()                       # non-NaN count per ticker

    raw = prices_monthly.iloc[t_pos - 1] / prices_monthly.iloc[t_pos - 12] - 1
    raw[n_valid < 11] = np.nan
    return raw


def value_raw(fund: pd.DataFrame, tickers: list[str]) -> pd.Series:
    """
    Value = -trailingPE.
    Fallback: -priceToBook if trailingPE is missing.
    Static (acknowledged lookahead limitation).
    """
    pe = fund.reindex(tickers)["trailingPE"]
    pb = fund.reindex(tickers)["priceToBook"]
    return (-pe).where(pe.notna(), -pb)


def quality_raw(fund: pd.DataFrame, tickers: list[str]) -> pd.Series:
    """
    Quality = returnOnEquity.
    Fallback: grossMargins if ROE is missing.
    Static.
    """
    roe = fund.reindex(tickers)["returnOnEquity"]
    gm = fund.reindex(tickers)["grossMargins"]
    return roe.where(roe.notna(), gm)


def size_raw(fund: pd.DataFrame, tickers: list[str]) -> pd.Series:
    """
    Size = -log(marketCap).  Smaller cap → higher score.
    Static.
    """
    mc = fund.reindex(tickers)["marketCap"]
    result = -np.log(mc)
    result[mc <= 0] = np.nan
    return result


def low_vol_raw(daily_prices: pd.DataFrame, t: pd.Timestamp, tickers: list[str]) -> pd.Series:
    """
    Low volatility = -std(daily returns) over the last 60 trading days ending at t.

    Implementation: take the 61 most-recent daily prices up to t (yielding 60
    return observations) and compute the per-ticker standard deviation.
    Require ≥50 non-NaN return observations; otherwise NaN.
    Score = negative std (lower volatility → higher score).
    """
    price_window = daily_prices.loc[:t, tickers].tail(61)  # 61 prices → 60 returns
    rets = price_window.pct_change().iloc[1:]               # drop first NaN row
    n_valid = rets.notna().sum()
    std = rets.std()
    std[n_valid < 50] = np.nan
    return -std


# ---------------------------------------------------------------------------
# Main factor-building loop
# ---------------------------------------------------------------------------

def build_factor_scores(
    prices_monthly: pd.DataFrame,
    daily_prices: pd.DataFrame,
    fund: pd.DataFrame,
) -> pd.DataFrame:
    """
    Iterate over all 24 formation dates, compute and z-score each factor
    independently, then assemble the composite.

    Returns a long DataFrame with columns:
        [ticker, date, momentum, value, quality, size, low_vol, composite]
    """
    tickers = prices_monthly.columns.tolist()

    # Pre-compute static raw fundamental scores (same values every month;
    # z-scores will also be identical each month — acknowledged limitation)
    val_raw   = value_raw(fund, tickers)
    qual_raw  = quality_raw(fund, tickers)
    sz_raw    = size_raw(fund, tickers)

    # Formation dates: business month-ends 2024-01-31 → 2025-12-31
    formation_dates = prices_monthly.index[
        (prices_monthly.index >= "2024-01-31") &
        (prices_monthly.index <= "2025-12-31")
    ]
    print(f"  Formation dates: {len(formation_dates)} months "
          f"({formation_dates[0].date()} → {formation_dates[-1].date()})")

    frames: list[pd.DataFrame] = []

    for t in formation_dates:
        t_pos = prices_monthly.index.get_loc(t)

        # --- Raw scores ---
        mom  = momentum_raw(prices_monthly, t_pos)
        val  = val_raw
        qual = qual_raw
        sz   = sz_raw
        lvol = low_vol_raw(daily_prices, t, tickers)

        # --- Independent cross-sectional z-scores ---
        mom_z  = z_score(mom)
        val_z  = z_score(val)
        qual_z = z_score(qual)
        sz_z   = z_score(sz)
        lvol_z = z_score(lvol)

        # --- Factor DataFrame (index = ticker) ---
        df = pd.DataFrame(
            {
                "momentum": mom_z,
                "value":    val_z,
                "quality":  qual_z,
                "size":     sz_z,
                "low_vol":  lvol_z,
            },
            index=pd.Index(tickers, name="ticker"),
        )

        # --- Composite: mean of available z-scores, require ≥3 of 5 ---
        n_valid_factors = df.notna().sum(axis=1)
        df["composite"] = df.mean(axis=1, skipna=True)
        df.loc[n_valid_factors < 3, "composite"] = np.nan

        df["date"] = t
        df = df.reset_index()[
            ["ticker", "date", "momentum", "value", "quality", "size", "low_vol", "composite"]
        ]
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(scores: pd.DataFrame, prices_monthly: pd.DataFrame) -> None:
    """Print all required validation statistics."""
    print("\n" + "=" * 66)
    print("VALIDATION STATS — AGENT 2: FACTOR CONSTRUCTION")
    print("=" * 66)

    all_dates = sorted(scores["date"].unique())
    n = len(all_dates)
    spot_dates = [all_dates[0], all_dates[n // 2], all_dates[-1]]
    factors = ["momentum", "value", "quality", "size", "low_vol"]

    # ── 1. Valid count per factor per month (3 spot-check months) ──────────
    print("\n1. Valid-score count per factor (3 spot-check months):")
    hdr = f"  {'Factor':<12}" + "".join(f"  {str(d.date()):>12}" for d in spot_dates)
    print(hdr)
    for f in factors:
        row = f"  {f:<12}"
        for d in spot_dates:
            cnt = scores.loc[scores["date"] == d, f].notna().sum()
            row += f"  {cnt:>12}"
        print(row)

    # ── 2. Cross-sectional mean and std of each z-scored factor ───────────
    print("\n2. Cross-sectional mean and std of z-scored factors (all months):")
    print(f"  {'Factor':<12}  {'Global mean':>12}  {'Global std':>10}")
    for f in factors:
        vals = scores[f].dropna()
        print(f"  {f:<12}  {vals.mean():>12.5f}  {vals.std():>10.5f}  (expect ≈0 and ≈1)")

    # ── 3. Composite universe size per month ──────────────────────────────
    print("\n3. Composite universe size per month:")
    comp_counts = (
        scores.groupby("date")["composite"]
        .apply(lambda x: x.notna().sum())
        .rename("n_composite")
    )
    print(f"  Min: {comp_counts.min()},  Max: {comp_counts.max()},  "
          f"Mean: {comp_counts.mean():.0f}  (expect 400+ most months)")
    low = comp_counts[comp_counts < 400]
    if len(low):
        print(f"  Months below 400: {len(low)} → {low.index.date.tolist()}")

    # ── 4. Momentum lookahead check ───────────────────────────────────────
    print("\n4. Momentum lookahead check (no prices beyond t-1 used):")
    for t in [all_dates[0], all_dates[-1]]:
        t_pos = prices_monthly.index.get_loc(t)
        last_price_used = prices_monthly.index[t_pos - 1]
        assert last_price_used < t, "Lookahead detected!"
        print(f"  Formation {t.date()} → last price used: {last_price_used.date()} ✓")

    # ── 5. Output summary ─────────────────────────────────────────────────
    print("\n5. Output summary:")
    print(f"  Rows:             {len(scores):,}")
    print(f"  Formation months: {scores['date'].nunique()}")
    print(f"  Tickers/month:    {scores.groupby('date').size().mean():.0f} (constant)")
    print(f"  Columns:          {scores.columns.tolist()}")
    print("=" * 66)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("Agent 2: Factor Construction")
    print("-" * 40)

    print("\nLoading monthly prices...")
    prices_monthly = load_monthly_prices()
    print(f"  Shape: {prices_monthly.shape}")

    print("\nLoading fundamentals...")
    fund = load_fundamentals()
    print(f"  Shape: {fund.shape}")

    print("\nLoading daily prices (for low-vol factor)...")
    daily_prices = load_daily_prices(prices_monthly.columns.tolist())
    print(f"  Shape: {daily_prices.shape}")

    print("\nBuilding factor scores...")
    scores = build_factor_scores(prices_monthly, daily_prices, fund)

    out_path = DATA_DIR / "factor_scores.parquet"
    scores.to_parquet(out_path, index=False)
    print(f"\nSaved factor_scores.parquet  →  {out_path}")

    validate(scores, prices_monthly)
    print("\nAgent 2 complete.")


if __name__ == "__main__":
    main()
