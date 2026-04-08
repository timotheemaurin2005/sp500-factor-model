"""
Agent 1: Data Ingestion
Fetches S&P 500 tickers, downloads price history, and collects fundamental data.
"""
import io
import os
import time
from pathlib import Path

import certifi
import numpy as np
import pandas as pd
import requests
import yfinance as yf

# Resolve paths relative to this file so the script works regardless of cwd
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def fetch_tickers() -> list[str]:
    """Pull S&P 500 tickers from Wikipedia and normalise to Yahoo Finance format."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, verify=certifi.where(), timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    tickers = tables[0]["Symbol"].tolist()
    # Replace '.' with '-' (e.g. BRK.B → BRK-B)
    tickers = [t.replace(".", "-") for t in tickers]
    print(f"  Loaded {len(tickers)} tickers from Wikipedia.")
    return tickers


def download_prices(tickers: list[str]) -> pd.DataFrame:
    """Download daily adjusted close prices in batches, resample to business month-end."""
    batch_size = 50
    all_closes: list[pd.DataFrame] = []

    n_batches = (len(tickers) + batch_size - 1) // batch_size
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        batch_num = i // batch_size + 1
        print(f"  Downloading batch {batch_num}/{n_batches} ({len(batch)} tickers)...")
        try:
            raw = yf.download(
                batch,
                start="2023-01-01",
                end="2026-02-01",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            # yfinance returns MultiIndex columns (field, ticker) for multiple tickers
            if isinstance(raw.columns, pd.MultiIndex):
                closes = raw["Close"]
            else:
                # Single ticker – columns are field names
                closes = raw[["Close"]].rename(columns={"Close": batch[0]})

            all_closes.append(closes)
        except Exception as exc:
            print(f"    Warning: batch {batch_num} failed ({exc})")
        time.sleep(0.1)

    if not all_closes:
        raise RuntimeError("No price data was downloaded.")

    # Concatenate all batches and deduplicate columns
    prices_daily = pd.concat(all_closes, axis=1)
    prices_daily = prices_daily.loc[:, ~prices_daily.columns.duplicated()]

    # Resample to business month-end (last trading day of each month)
    prices_monthly = prices_daily.resample("BME").last()

    # Drop tickers with >20% missing monthly observations
    min_obs = int(len(prices_monthly) * 0.80)
    n_before = prices_monthly.shape[1]
    prices_monthly = prices_monthly.dropna(axis=1, thresh=min_obs)
    n_dropped = n_before - prices_monthly.shape[1]

    print(
        f"  {prices_monthly.shape[1]} tickers retained, "
        f"{n_dropped} dropped (>20% missing)."
    )
    return prices_monthly


def fetch_fundamentals(tickers: list[str]) -> pd.DataFrame:
    """Fetch fundamental fields from yfinance .info for each ticker."""
    fields = ["trailingPE", "priceToBook", "returnOnEquity", "grossMargins", "marketCap"]
    records = []

    for idx, ticker in enumerate(tickers):
        if (idx + 1) % 50 == 0:
            print(f"  Fetched fundamentals for {idx + 1}/{len(tickers)} tickers...")
        try:
            info = yf.Ticker(ticker).info
            record = {"ticker": ticker}
            for field in fields:
                record[field] = info.get(field, np.nan)
        except Exception:
            record = {"ticker": ticker, **{f: np.nan for f in fields}}
        records.append(record)

    return pd.DataFrame(records)


def validate(prices: pd.DataFrame, fundamentals: pd.DataFrame, n_dropped: int) -> None:
    """Print validation statistics."""
    print("\n" + "=" * 52)
    print("VALIDATION STATS")
    print("=" * 52)
    print(f"Prices shape:             {prices.shape}  (months × tickers)")
    print(
        f"Prices date range:        "
        f"{prices.index.min().strftime('%Y-%m-%d')} → "
        f"{prices.index.max().strftime('%Y-%m-%d')}"
    )
    print(f"Tickers retained:         {prices.shape[1]}")
    print(f"Tickers dropped (>20%):   {n_dropped}")

    valid_fund = (
        fundamentals[["trailingPE", "priceToBook", "returnOnEquity", "grossMargins", "marketCap"]]
        .notna()
        .any(axis=1)
        .sum()
    )
    print(f"Tickers with ≥1 fundamental field: {valid_fund}")

    print("\nFirst 5 index values (should all be business month-end):")
    for d in prices.index[:5]:
        print(f"  {d.strftime('%Y-%m-%d')} ({d.day_name()})")
    print("=" * 52)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Agent 1: Data Ingestion")
    print("-" * 40)

    print("\nStep 1 — Ticker list")
    tickers = fetch_tickers()

    print("\nStep 2 — Price data")
    prices = download_prices(tickers)
    n_dropped = len(tickers) - prices.shape[1]
    prices.to_parquet(DATA_DIR / "prices.parquet")
    print(f"  Saved prices.parquet  →  {DATA_DIR / 'prices.parquet'}")

    print("\nStep 3 — Fundamental data")
    fundamentals = fetch_fundamentals(tickers)
    fundamentals.to_parquet(DATA_DIR / "fundamentals.parquet", index=False)
    print(f"  Saved fundamentals.parquet  →  {DATA_DIR / 'fundamentals.parquet'}")

    validate(prices, fundamentals, n_dropped)
    print("\nAgent 1 complete.")


if __name__ == "__main__":
    main()
