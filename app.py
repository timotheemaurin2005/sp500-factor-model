"""
S&P 500 Multi-Factor Model — Streamlit App
Visualises five equity factors (Momentum, Value, Quality, Size, Low Volatility)
backtested over the 2024-2025 evaluation period.

All data is loaded from pre-generated parquet files in data/.
No yfinance calls are made at render time.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="S&P 500 Factor Model",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).parent / "data"

FACTORS = ["momentum", "value", "quality", "size", "low_vol", "composite"]
FACTOR_LABELS = {
    "momentum": "Momentum",
    "value":    "Value",
    "quality":  "Quality",
    "size":     "Size",
    "low_vol":  "Low Volatility",
    "composite":"Composite",
}
# Tableau 10 palette
FACTOR_COLORS = {
    "momentum":  "#4E79A7",
    "value":     "#F28E2B",
    "quality":   "#59A14F",
    "size":      "#E15759",
    "low_vol":   "#76B7B2",
    "composite": "#B07AA1",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#FAFAFA", family="Inter, sans-serif"),
    xaxis=dict(gridcolor="#333", zerolinecolor="#555"),
    yaxis=dict(gridcolor="#333", zerolinecolor="#555"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#444"),
    margin=dict(l=40, r=20, t=50, b=40),
)


# ---------------------------------------------------------------------------
# Data loading — all cached
# ---------------------------------------------------------------------------

@st.cache_data
def load_backtest_results() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "backtest_results.parquet")


@st.cache_data
def load_portfolio_returns() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "portfolio_returns.parquet")


@st.cache_data
def load_factor_scores() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "factor_scores.parquet")


@st.cache_data
def load_fundamentals() -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / "fundamentals.parquet")
    return df.set_index("ticker")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough decline in a cumulative return series."""
    cum = (1.0 + returns).cumprod()
    rolling_peak = cum.cummax()
    dd = (cum - rolling_peak) / rolling_peak
    return float(dd.min())


def annualised_return(returns: pd.Series) -> float:
    return float(returns.mean() * 12)


def annualised_vol(returns: pd.Series) -> float:
    return float(returns.std() * np.sqrt(12))


def sharpe(returns: pd.Series, rf: float = 0.05) -> float:
    ann_r = annualised_return(returns)
    ann_v = annualised_vol(returns)
    return (ann_r - rf) / ann_v if ann_v > 0 else np.nan


def cumulative_returns(returns: pd.Series) -> pd.Series:
    """Cumulative return series starting from 0."""
    return (1.0 + returns).cumprod() - 1.0


def assign_quintiles(scores: pd.Series) -> pd.Series:
    """Q1=lowest 20%, Q5=highest 20%. NaN for missing scores."""
    valid = scores.dropna()
    if len(valid) < 5:
        return pd.Series(np.nan, index=scores.index)
    ranked = valid.rank(method="first")
    quints = pd.qcut(ranked, 5, labels=[1, 2, 3, 4, 5]).astype(float)
    return quints.reindex(scores.index)


def fmt_pct(v: float, decimals: int = 1) -> str:
    return f"{v:.{decimals}%}" if pd.notna(v) else "N/A"


def fmt_num(v: float, decimals: int = 2) -> str:
    return f"{v:.{decimals}f}" if pd.notna(v) else "N/A"


def fmt_mcap(v: float) -> str:
    if pd.isna(v):
        return "N/A"
    if v >= 1e12:
        return f"${v/1e12:.1f}T"
    if v >= 1e9:
        return f"${v/1e9:.1f}B"
    return f"${v/1e6:.0f}M"


# ---------------------------------------------------------------------------
# Guard: require all data files
# ---------------------------------------------------------------------------
REQUIRED_FILES = [
    "backtest_results.parquet",
    "fundamentals.parquet",
    "factor_scores.parquet",
    "portfolio_returns.parquet",
    "prices.parquet",
]
missing = [f for f in REQUIRED_FILES if not (DATA_DIR / f).exists()]
if missing:
    st.error("Data files not found. Run agents 1–3 first.")
    st.stop()


# ---------------------------------------------------------------------------
# Load all data
# ---------------------------------------------------------------------------
backtest   = load_backtest_results().set_index("factor")
port_ret   = load_portfolio_returns()
scores_all = load_factor_scores()
fund       = load_fundamentals()


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("S&P 500 Factor Model")
st.sidebar.markdown("---")
section = st.sidebar.radio(
    "Navigate",
    [
        "About",
        "Factor Performance",
        "Cumulative Returns",
        "Quintile Breakdown",
        "Stock Explorer",
        "Composite Portfolio",
    ],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.caption("Data: 2024–2025  |  Universe: S&P 500")


# ===========================================================================
# Section 1 — About
# ===========================================================================
if section == "About":
    st.title("S&P 500 Multi-Factor Model")
    st.markdown("**Evaluation Period: 2024 – 2025  |  Universe: ~498 S&P 500 Stocks**")

    st.markdown(
        """
A **factor model** scores stocks along characteristics that have historically
been associated with excess returns — such as recent price momentum, cheap
valuation, high profitability, small market capitalisation, or low volatility.
Each month, stocks are sorted by their factor score, grouped into quintiles
(Q1 = weakest, Q5 = strongest), and held equally-weighted for one month.
The long-short spread (Q5 minus Q1) measures how well the factor separates
winners from losers.

This project applies five classic equity factors to today's S&P 500 universe
over the 2024–2025 period.  Monthly factor scores are constructed at each
business-month-end rebalance date.  **Momentum** uses a 12-1 skip-month
calculation to avoid short-term reversal contamination.  **Value** is
proxied by the negative trailing P/E (falling back to P/B where P/E is
unavailable).  **Quality** uses return on equity (falling back to gross
margins).  **Size** uses the negative log of market capitalisation.
**Low Volatility** uses the negative 60-trading-day daily return standard
deviation.  Each factor is z-scored cross-sectionally and independently each
month, and a **Composite** score averages any available factor z-scores
(requiring at least 3 of 5).
"""
    )

    st.markdown("#### Methodology at a Glance")
    col1, col2, col3 = st.columns(3)
    col1.metric("Factors", "5 + Composite")
    col2.metric("Rebalance Frequency", "Monthly")
    col3.metric("Evaluation Period", "2024 – 2025")
    col1.metric("Portfolio Construction", "Equal-Weight Quintiles")
    col2.metric("Momentum Convention", "12-1 Skip-Month")
    col3.metric("Universe", "~498 S&P 500 Stocks")

    st.markdown("---")
    st.warning(
        """
**Data & Methodology Limitations — Please Read**

1. **Survivorship bias:** The constituent list is today's S&P 500 pulled from
   Wikipedia.  Stocks delisted or removed during 2024–2025 are excluded,
   which biases results upward relative to a live strategy.

2. **Fundamental lookahead bias:** Trailing P/E, P/B, ROE, and gross margins
   are current values fetched from yfinance, *not* point-in-time historical
   values.  The Value, Quality, and Size factors are best understood as a
   static cross-sectional ranking exercise, not a true historical simulation.

3. **Static market cap:** The Size factor uses today's market capitalisation
   for all 24 rebalance months.  Market caps shift materially over two years.

4. **Benchmark is equal-weighted, not cap-weighted:** The benchmark shown
   throughout this app is the equal-weighted average monthly return across
   all universe stocks — it is **not** a cap-weighted S&P 500 / SPY-like
   index.  Do not compare directly against SPY performance.

*This project is for educational and portfolio-showcase purposes only.
It is not investment advice.*
"""
    )


# ===========================================================================
# Section 2 — Factor Performance
# ===========================================================================
elif section == "Factor Performance":
    st.title("Factor Performance")
    st.caption(
        "Annualised Q5−Q1 long-short spread returns over 2024–2025.  "
        "Equal-weight quintiles, monthly rebalance."
    )

    # ── Horizontal bar chart ─────────────────────────────────────────────
    chart_df = backtest["ann_return"].rename("ann_return").reset_index()
    chart_df = chart_df.sort_values("ann_return", ascending=True)
    colors = [FACTOR_COLORS[f] for f in chart_df["factor"]]
    labels = [FACTOR_LABELS[f] for f in chart_df["factor"]]

    fig_bar = go.Figure(go.Bar(
        x=chart_df["ann_return"],
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[fmt_pct(v) for v in chart_df["ann_return"]],
        textposition="outside",
    ))
    fig_bar.add_vline(x=0, line_color="#888", line_dash="dash")
    fig_bar.update_layout(
        **PLOTLY_LAYOUT,
        title="Annualised Q5−Q1 Return by Factor",
        xaxis_title="Annualised Return",
        yaxis_title="",
        height=380,
        xaxis_tickformat=".0%",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Summary table ────────────────────────────────────────────────────
    st.markdown("#### Performance Summary")

    def sharpe_color(val):
        if pd.isna(val):
            return ""
        if val > 0.5:
            return "color: #59A14F"   # green
        if val >= 0:
            return "color: #F28E2B"   # yellow/orange
        return "color: #E15759"       # red

    table = pd.DataFrame({
        "Factor":          [FACTOR_LABELS[f] for f in FACTORS],
        "Ann Return":      [fmt_pct(backtest.loc[f, "ann_return"]) for f in FACTORS],
        "Ann Vol":         [fmt_pct(backtest.loc[f, "ann_vol"]) for f in FACTORS],
        "Sharpe":          [backtest.loc[f, "sharpe"] for f in FACTORS],
        "Max Drawdown":    [fmt_pct(backtest.loc[f, "max_drawdown"]) for f in FACTORS],
        "Hit Rate":        [fmt_pct(backtest.loc[f, "hit_rate"]) for f in FACTORS],
        "Q5 One-Way TO":   [fmt_pct(backtest.loc[f, "q5_one_way_turnover"]) for f in FACTORS],
        "L/S Combined TO": [fmt_pct(backtest.loc[f, "ls_combined_turnover"]) for f in FACTORS],
    })

    styled = (
        table.style
        .map(sharpe_color, subset=["Sharpe"])
        .format({"Sharpe": lambda v: fmt_num(v, 3)})
        .hide(axis="index")
    )
    st.dataframe(styled, use_container_width=True, height=280)

    st.caption(
        "**Q5 One-Way TO** = fraction of Q5 portfolio replaced each rebalance.  "
        "**L/S Combined TO** = average of Q5 and Q1 one-way turnovers.  "
        "Static fundamental factors (Value, Quality, Size) show 0% turnover "
        "because ranks are fixed across all months (acknowledged limitation)."
    )


# ===========================================================================
# Section 3 — Cumulative Returns
# ===========================================================================
elif section == "Cumulative Returns":
    st.title("Cumulative Returns")
    st.caption(
        "Cumulative Q5−Q1 long-short spread return for each factor over 2024–2025.  "
        "Click legend entries to toggle factors."
    )

    fig_cum = go.Figure()

    for f in FACTORS:
        spread = port_ret[f"{f}_spread"].dropna()
        cum = cumulative_returns(spread)
        # Prepend a base point at 0
        base_idx = pd.DatetimeIndex([spread.index[0] - pd.DateOffset(months=1)])
        cum_full = pd.concat([pd.Series([0.0], index=base_idx), cum])

        fig_cum.add_trace(go.Scatter(
            x=cum_full.index,
            y=cum_full.values,
            mode="lines",
            name=FACTOR_LABELS[f],
            line=dict(color=FACTOR_COLORS[f], width=2.5),
        ))

    # Dashed zero line
    fig_cum.add_hline(y=0, line_color="#888", line_dash="dash", line_width=1)

    fig_cum.update_layout(
        **PLOTLY_LAYOUT,
        title="Cumulative Q5−Q1 Long-Short Return",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        yaxis_tickformat=".0%",
        height=500,
        hovermode="x unified",
    )
    st.plotly_chart(fig_cum, use_container_width=True)

    # Summary stats below chart
    st.markdown("#### Final Cumulative Returns")
    cols = st.columns(len(FACTORS))
    for i, f in enumerate(FACTORS):
        spread = port_ret[f"{f}_spread"].dropna()
        final_cum = cumulative_returns(spread).iloc[-1]
        cols[i].metric(FACTOR_LABELS[f], fmt_pct(final_cum))


# ===========================================================================
# Section 4 — Quintile Breakdown
# ===========================================================================
elif section == "Quintile Breakdown":
    st.title("Quintile Breakdown")

    factor_choice = st.selectbox(
        "Select factor",
        FACTORS,
        format_func=lambda f: FACTOR_LABELS[f],
    )

    q1_ann  = annualised_return(port_ret[f"{factor_choice}_Q1"].dropna())
    q3_ann  = annualised_return(port_ret[f"{factor_choice}_Q3"].dropna())
    q5_ann  = annualised_return(port_ret[f"{factor_choice}_Q5"].dropna())
    bm_ann  = annualised_return(port_ret["benchmark"].dropna())

    quintiles   = ["Q1 (Lowest)", "Q3 (Middle)", "Q5 (Highest)"]
    ann_returns = [q1_ann, q3_ann, q5_ann]
    bar_colors  = ["#E15759", "#F28E2B", "#59A14F"]

    fig_q = go.Figure(go.Bar(
        x=quintiles,
        y=ann_returns,
        marker_color=bar_colors,
        text=[fmt_pct(v) for v in ann_returns],
        textposition="outside",
        width=0.4,
    ))
    # Benchmark reference line
    fig_q.add_hline(
        y=bm_ann,
        line_color="#76B7B2",
        line_dash="dash",
        line_width=1.5,
        annotation_text=f"Equal-Weighted Universe Benchmark ({fmt_pct(bm_ann)})",
        annotation_position="top right",
        annotation_font_color="#76B7B2",
    )
    fig_q.add_hline(y=0, line_color="#888", line_dash="dot", line_width=1)
    fig_q.update_layout(
        **PLOTLY_LAYOUT,
        title=f"{FACTOR_LABELS[factor_choice]} — Annualised Return by Quintile",
        xaxis_title="Quintile",
        yaxis_title="Annualised Return",
        yaxis_tickformat=".0%",
        height=420,
        showlegend=False,
    )
    st.plotly_chart(fig_q, use_container_width=True)

    spread_ann = q5_ann - q1_ann
    st.caption(
        f"**Q1→Q5 monotonic spread: {fmt_pct(spread_ann)} annualised.**  "
        "A factor works as expected when Q5 consistently outperforms Q1 and returns "
        "increase monotonically from Q1 to Q5.  A flat or inverted pattern suggests "
        "the factor has no predictive power in this period."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Q1 Ann Return", fmt_pct(q1_ann))
    col2.metric("Q3 Ann Return", fmt_pct(q3_ann))
    col3.metric("Q5 Ann Return", fmt_pct(q5_ann))
    col4.metric("Q5−Q1 Spread", fmt_pct(spread_ann))


# ===========================================================================
# Section 5 — Stock Explorer
# ===========================================================================
elif section == "Stock Explorer":
    st.title("Stock Explorer")
    st.caption(
        "Factor z-scores and quintile ranks as of the most recent formation date "
        "(2025-12-31).  Z-scores are cross-sectional — a score of +1 means the "
        "stock is 1 standard deviation above the cross-sectional mean."
    )

    all_tickers = sorted(scores_all["ticker"].unique())
    ticker = st.selectbox("Select ticker", all_tickers)

    # Scores at last formation date
    last_date = scores_all["date"].max()
    scores_last = scores_all[scores_all["date"] == last_date].set_index("ticker")

    PRICE_FACTORS = ["momentum", "value", "quality", "size", "low_vol"]

    # Compute quintile ranks for each factor at last date
    quintile_ranks = {}
    for f in PRICE_FACTORS:
        quints = assign_quintiles(scores_last[f])
        quintile_ranks[f] = quints

    ticker_scores = scores_last.loc[ticker] if ticker in scores_last.index else None

    if ticker_scores is None:
        st.warning(f"{ticker} has no factor scores at {last_date.date()}.")
    else:
        # ── Factor score table ───────────────────────────────────────────
        st.markdown(f"#### {ticker} — Factor Scores at {last_date.date()}")
        rows = []
        for f in PRICE_FACTORS:
            z = ticker_scores[f]
            q = quintile_ranks[f].get(ticker, np.nan)
            rows.append({
                "Factor":      FACTOR_LABELS[f],
                "Z-Score":     round(float(z), 3) if pd.notna(z) else None,
                "Quintile":    int(q) if pd.notna(q) else None,
                "Interpretation": (
                    f"Q{int(q)} of 5" if pd.notna(q) else "Insufficient data"
                ),
            })
        comp_z = ticker_scores["composite"]
        rows.append({
            "Factor": "Composite",
            "Z-Score": round(float(comp_z), 3) if pd.notna(comp_z) else None,
            "Quintile": None,
            "Interpretation": "Mean of available factor z-scores",
        })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # ── Radar chart ─────────────────────────────────────────────────
        radar_factors = PRICE_FACTORS
        radar_labels  = [FACTOR_LABELS[f] for f in radar_factors]
        radar_values  = [
            float(ticker_scores[f]) if pd.notna(ticker_scores[f]) else 0.0
            for f in radar_factors
        ]
        # Close the polygon
        radar_values_closed = radar_values + [radar_values[0]]
        radar_labels_closed = radar_labels + [radar_labels[0]]

        fig_radar = go.Figure(go.Scatterpolar(
            r=radar_values_closed,
            theta=radar_labels_closed,
            fill="toself",
            fillcolor=f"rgba(78, 121, 167, 0.3)",
            line=dict(color="#4E79A7", width=2),
            name=ticker,
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(
                    visible=True,
                    gridcolor="#444",
                    tickfont=dict(color="#FAFAFA"),
                    range=[
                        min(-2.5, min(radar_values) - 0.5),
                        max(2.5, max(radar_values) + 0.5),
                    ],
                ),
                angularaxis=dict(gridcolor="#444", tickfont=dict(color="#FAFAFA")),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#FAFAFA"),
            title=f"{ticker} — Factor Z-Score Profile",
            height=420,
            showlegend=False,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # ── Fundamentals metrics row ─────────────────────────────────────
        st.markdown("#### Fundamental Snapshot (Current Values)")
        if ticker in fund.index:
            row = fund.loc[ticker]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Trailing P/E",   fmt_num(row.get("trailingPE", np.nan), 1))
            c2.metric("Price / Book",   fmt_num(row.get("priceToBook", np.nan), 2))
            c3.metric("ROE",            fmt_pct(row.get("returnOnEquity", np.nan)))
            c4.metric("Market Cap",     fmt_mcap(row.get("marketCap", np.nan)))
            st.caption(
                "Note: these are current values from yfinance `.info`, "
                "not point-in-time historical figures (acknowledged limitation)."
            )
        else:
            st.caption(f"No fundamental data available for {ticker}.")


# ===========================================================================
# Section 6 — Composite Portfolio
# ===========================================================================
elif section == "Composite Portfolio":
    st.title("Composite Portfolio")
    st.caption(
        "Performance of the Composite Q5 portfolio (top quintile ranked by composite "
        "z-score) versus the Equal-Weighted Universe Benchmark over 2024–2025."
    )

    comp_q5_ret = port_ret["composite_Q5"].dropna()
    bm_ret      = port_ret["benchmark"].dropna()

    # Align to common dates
    common_idx = comp_q5_ret.index.intersection(bm_ret.index)
    comp_q5_ret = comp_q5_ret.loc[common_idx]
    bm_ret      = bm_ret.loc[common_idx]

    # Cumulative returns (prepend 0 base)
    base_idx    = pd.DatetimeIndex([common_idx[0] - pd.DateOffset(months=1)])
    cum_comp    = pd.concat([pd.Series([0.0], index=base_idx), cumulative_returns(comp_q5_ret)])
    cum_bm      = pd.concat([pd.Series([0.0], index=base_idx), cumulative_returns(bm_ret)])

    # ── Line chart ───────────────────────────────────────────────────────
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(
        x=cum_comp.index, y=cum_comp.values,
        mode="lines",
        name="Composite Q5",
        line=dict(color=FACTOR_COLORS["composite"], width=2.5),
    ))
    fig_comp.add_trace(go.Scatter(
        x=cum_bm.index, y=cum_bm.values,
        mode="lines",
        name="Equal-Weighted Universe Benchmark",
        line=dict(color="#888", width=2, dash="dash"),
    ))
    fig_comp.add_hline(y=0, line_color="#555", line_dash="dot", line_width=1)
    fig_comp.update_layout(
        **PLOTLY_LAYOUT,
        title="Composite Q5 vs Equal-Weighted Universe Benchmark",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        yaxis_tickformat=".0%",
        height=460,
        hovermode="x unified",
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    # ── Metric cards ─────────────────────────────────────────────────────
    comp_ann_r  = annualised_return(comp_q5_ret)
    comp_sharpe = sharpe(comp_q5_ret)
    comp_mdd    = max_drawdown(comp_q5_ret)

    bm_ann_r    = annualised_return(bm_ret)
    bm_sharpe   = sharpe(bm_ret)
    bm_mdd      = max_drawdown(bm_ret)

    st.markdown("#### Performance Metrics")
    st.markdown(
        "<small>Left value = Composite Q5 &nbsp;|&nbsp; "
        "Delta = Q5 minus Equal-Weighted Universe Benchmark</small>",
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Annualised Return",
        fmt_pct(comp_ann_r),
        delta=fmt_pct(comp_ann_r - bm_ann_r),
        help="Composite Q5 vs Equal-Weighted Universe Benchmark",
    )
    col2.metric(
        "Sharpe Ratio",
        fmt_num(comp_sharpe),
        delta=fmt_num(comp_sharpe - bm_sharpe),
        help="Risk-free rate = 5 % p.a.",
    )
    col3.metric(
        "Max Drawdown",
        fmt_pct(comp_mdd),
        delta=fmt_pct(comp_mdd - bm_mdd),
        delta_color="inverse",
        help="Less negative is better",
    )

    st.markdown("---")
    with st.expander("Benchmark definition"):
        st.markdown(
            "The **Equal-Weighted Universe Benchmark** is the simple average monthly "
            "return across all ~498 universe stocks with a valid price in that month.  "
            "It is **not** a cap-weighted index and should not be compared directly "
            "to SPY or any published S&P 500 index return.  It reflects the average "
            "performance of the survivorship-biased constituent list used throughout "
            "this model."
        )
