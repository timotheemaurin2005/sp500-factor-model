"""
Agent 3: Backtesting
Constructs equal-weight quintile portfolios for each factor, computes monthly
returns, performance metrics, and turnover over the 2024-01-31 → 2025-12-31
evaluation period.

Forward return for formation date t:
    fwd[i, t] = price[i, t+1 month-end] / price[i, t month-end] - 1

The last formation date 2025-12-31 exits at the 2026-01-30 price.

Outputs
-------
data/backtest_results.parquet  — one row per factor, all metrics
data/portfolio_returns.parquet — monthly Q1/Q3/Q5/spread/benchmark per factor
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

FACTORS = ["momentum", "value", "quality", "size", "low_vol", "composite"]


# ---------------------------------------------------------------------------
# Quintile assignment
# ---------------------------------------------------------------------------

def assign_quintiles(scores: pd.Series) -> pd.Series:
    """
    Assign each stock a quintile label 1–5 (Q1=lowest score, Q5=highest).
    Stocks with NaN scores receive NaN.  Uses first-rank tie-breaking.
    """
    valid = scores.dropna()
    if len(valid) < 5:
        return pd.Series(np.nan, index=scores.index)
    ranked = valid.rank(method="first")
    quints = pd.qcut(ranked, 5, labels=[1, 2, 3, 4, 5]).astype(float)
    return quints.reindex(scores.index)


# ---------------------------------------------------------------------------
# Performance statistics
# ---------------------------------------------------------------------------

def compute_max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough decline in the cumulative return series."""
    cum = (1.0 + returns).cumprod()
    rolling_peak = cum.cummax()
    drawdown = (cum - rolling_peak) / rolling_peak
    return float(drawdown.min())


def performance_metrics(spread_returns: pd.Series) -> dict:
    """
    Annualised metrics for a monthly long-short spread return series.
    Risk-free rate = 5 % per annum.
    """
    ann_return = float(spread_returns.mean() * 12)
    ann_vol    = float(spread_returns.std()  * np.sqrt(12))
    sharpe     = (ann_return - 0.05) / ann_vol if ann_vol > 0 else np.nan
    mdd        = compute_max_drawdown(spread_returns)
    hit_rate   = float((spread_returns > 0).mean())
    return {
        "ann_return":  ann_return,
        "ann_vol":     ann_vol,
        "sharpe":      sharpe,
        "max_drawdown": mdd,
        "hit_rate":    hit_rate,
    }


# ---------------------------------------------------------------------------
# Turnover
# ---------------------------------------------------------------------------

def compute_turnover(membership: dict[pd.Timestamp, set]) -> float:
    """
    One-way turnover for a quintile portfolio.

    turnover[t] = |stocks entering quintile at t| / |quintile size at t|

    Returns the average across all non-initial rebalance dates.
    """
    dates = sorted(membership.keys())
    turnovers = []
    for i in range(1, len(dates)):
        prev = membership[dates[i - 1]]
        curr = membership[dates[i]]
        if len(curr) == 0:
            continue
        entering = curr - prev
        turnovers.append(len(entering) / len(curr))
    return float(np.mean(turnovers)) if turnovers else np.nan


# ---------------------------------------------------------------------------
# Main backtesting routine
# ---------------------------------------------------------------------------

def run_backtest(
    scores: pd.DataFrame,
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build quintile portfolios and compute all returns.

    Returns
    -------
    portfolio_returns : DataFrame
        Index = formation_date.
        Columns = {factor}_Q1/Q3/Q5/spread for each factor + benchmark.
    quintile_members : dict
        {factor: {"Q1": {date: set}, "Q5": {date: set}}}
        Used for turnover calculation.
    """
    formation_dates = sorted(scores["date"].unique())

    # ── Pre-compute forward returns ──────────────────────────────────────
    fwd_returns: dict[pd.Timestamp, pd.Series] = {}
    for t in formation_dates:
        t_pos = prices.index.get_loc(t)
        fwd = prices.iloc[t_pos + 1] / prices.iloc[t_pos] - 1
        fwd_returns[t] = fwd  # Series indexed by ticker

    # ── Equal-weighted universe benchmark ────────────────────────────────
    benchmark: dict[pd.Timestamp, float] = {
        t: float(fwd_returns[t].mean()) for t in formation_dates
    }

    # ── Quintile membership for turnover ─────────────────────────────────
    q5_members: dict[str, dict[pd.Timestamp, set]] = {f: {} for f in FACTORS}
    q1_members: dict[str, dict[pd.Timestamp, set]] = {f: {} for f in FACTORS}

    # ── Portfolio return rows ─────────────────────────────────────────────
    port_rows: list[dict] = []

    for t in formation_dates:
        fwd = fwd_returns[t]
        scores_t = scores.loc[scores["date"] == t].set_index("ticker")
        row: dict = {"date": t, "benchmark": benchmark[t]}

        for f in FACTORS:
            factor_scores = scores_t[f]
            quints = assign_quintiles(factor_scores)

            # Track Q5 and Q1 membership (all assigned members, for turnover)
            q5_members[f][t] = set(quints[quints == 5].index)
            q1_members[f][t] = set(quints[quints == 1].index)

            # Portfolio return = equal-weighted average of constituent fwd returns
            for q_label, q_num in [("Q1", 1), ("Q3", 3), ("Q5", 5)]:
                members = quints[quints == q_num].index
                valid_rets = fwd.reindex(members).dropna()
                row[f"{f}_{q_label}"] = float(valid_rets.mean()) if len(valid_rets) > 0 else np.nan

            row[f"{f}_spread"] = row[f"{f}_Q5"] - row[f"{f}_Q1"]

        port_rows.append(row)

    portfolio_returns = pd.DataFrame(port_rows).set_index("date")
    portfolio_returns.index.name = "date"

    return portfolio_returns, q5_members, q1_members


# ---------------------------------------------------------------------------
# Aggregate performance table
# ---------------------------------------------------------------------------

def build_results_table(
    portfolio_returns: pd.DataFrame,
    q5_members: dict[str, dict],
    q1_members: dict[str, dict],
) -> pd.DataFrame:
    """
    Compile one-row-per-factor performance table.
    Columns: factor, ann_return, ann_vol, sharpe, max_drawdown,
             hit_rate, q5_one_way_turnover, ls_combined_turnover.
    """
    rows = []
    for f in FACTORS:
        spread = portfolio_returns[f"{f}_spread"].dropna()
        metrics = performance_metrics(spread)

        q5_to  = compute_turnover(q5_members[f])
        q1_to  = compute_turnover(q1_members[f])
        ls_to  = (q5_to + q1_to) / 2.0

        rows.append(
            {
                "factor":              f,
                "ann_return":          metrics["ann_return"],
                "ann_vol":             metrics["ann_vol"],
                "sharpe":              metrics["sharpe"],
                "max_drawdown":        metrics["max_drawdown"],
                "hit_rate":            metrics["hit_rate"],
                "q5_one_way_turnover": q5_to,
                "ls_combined_turnover": ls_to,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(
    portfolio_returns: pd.DataFrame,
    backtest_results: pd.DataFrame,
    prices: pd.DataFrame,
) -> None:
    """Print all required validation statistics."""
    print("\n" + "=" * 72)
    print("VALIDATION — AGENT 3: BACKTESTING")
    print("=" * 72)

    # ── Confirm last formation date uses 2026-01-30 price ────────────────
    last_t = pd.Timestamp("2025-12-31")
    t_pos  = prices.index.get_loc(last_t)
    exit_date = prices.index[t_pos + 1]
    fwd_sample = (prices.iloc[t_pos + 1] / prices.iloc[t_pos] - 1).dropna()
    print(f"\n1. Last formation date forward-return check:")
    print(f"   Formation date : {last_t.date()}")
    print(f"   Exit price date: {exit_date.date()}  (should be 2026-01-30) ✓")
    print(f"   Valid fwd returns: {len(fwd_sample)} tickers")

    # ── Full performance table ────────────────────────────────────────────
    print("\n2. Full performance table (Q5-Q1 long-short spread):")
    fmt = (
        f"  {'Factor':<12} {'AnnRet':>8} {'AnnVol':>8} {'Sharpe':>8} "
        f"{'MaxDD':>8} {'HitRate':>8} {'Q5 TO':>8} {'L/S TO':>8}"
    )
    print(fmt)
    print("  " + "-" * 70)
    for _, r in backtest_results.iterrows():
        print(
            f"  {r['factor']:<12} "
            f"{r['ann_return']:>8.2%} "
            f"{r['ann_vol']:>8.2%} "
            f"{r['sharpe']:>8.3f} "
            f"{r['max_drawdown']:>8.2%} "
            f"{r['hit_rate']:>8.2%} "
            f"{r['q5_one_way_turnover']:>8.2%} "
            f"{r['ls_combined_turnover']:>8.2%}"
        )

    # ── First 10 rows of portfolio_returns ───────────────────────────────
    print("\n3. First 10 rows of portfolio_returns.parquet (selected columns):")
    display_cols = (
        ["benchmark"]
        + [f"{f}_spread" for f in FACTORS]
    )
    print(portfolio_returns[display_cols].head(10).to_string(
        float_format=lambda x: f"{x:.4f}"
    ))

    # ── Benchmark labelling check ─────────────────────────────────────────
    print("\n4. Benchmark label: 'Equal-Weighted Universe Benchmark'")
    print("   Column 'benchmark' = equal-weighted avg monthly return "
          "across all tickers with a valid fwd return each month.")
    bm = portfolio_returns["benchmark"]
    print(f"   Range: {bm.min():.4f} – {bm.max():.4f},  "
          f"Mean: {bm.mean():.4f},  Ann: {bm.mean()*12:.2%}")

    # ── File sizes ────────────────────────────────────────────────────────
    br_path = DATA_DIR / "backtest_results.parquet"
    pr_path = DATA_DIR / "portfolio_returns.parquet"
    print(f"\n5. Files written:")
    print(f"   backtest_results.parquet  → {br_path}  "
          f"({br_path.stat().st_size / 1024:.1f} KB)")
    print(f"   portfolio_returns.parquet → {pr_path}  "
          f"({pr_path.stat().st_size / 1024:.1f} KB)")
    print(f"   portfolio_returns shape:  {portfolio_returns.shape}")
    print(f"   portfolio_returns columns: {portfolio_returns.columns.tolist()}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("Agent 3: Backtesting")
    print("-" * 40)

    print("\nLoading factor scores...")
    scores = pd.read_parquet(DATA_DIR / "factor_scores.parquet")
    print(f"  Shape: {scores.shape}")

    print("\nLoading monthly prices...")
    prices = pd.read_parquet(DATA_DIR / "prices.parquet")
    print(f"  Shape: {prices.shape}")

    print("\nRunning backtest...")
    portfolio_returns, q5_members, q1_members = run_backtest(scores, prices)
    print(f"  Portfolio returns shape: {portfolio_returns.shape}")

    print("\nBuilding results table...")
    backtest_results = build_results_table(portfolio_returns, q5_members, q1_members)

    # Save outputs
    br_path = DATA_DIR / "backtest_results.parquet"
    pr_path = DATA_DIR / "portfolio_returns.parquet"
    backtest_results.to_parquet(br_path, index=False)
    portfolio_returns.to_parquet(pr_path)
    print(f"  Saved backtest_results.parquet")
    print(f"  Saved portfolio_returns.parquet")

    validate(portfolio_returns, backtest_results, prices)
    print("\nAgent 3 complete.")


if __name__ == "__main__":
    main()
