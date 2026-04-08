# S&P500 Factor Model — Project Instructions (Final)

## Project Overview
Build a deployable S&P500 multi-factor model that constructs, backtests, and visualises five equity factors
(Momentum, Value, Quality, Size, Low Volatility) over the evaluation period 2024–2025.
Output is a public Streamlit web app deployed on Streamlit Community Cloud, linkable from GitHub.

---

## Known Limitations (acknowledge in README and app disclaimer)
1. **Survivorship bias** — constituent list is today's S&P500 from Wikipedia. Stocks delisted or removed during 2024–2025 are excluded, which biases results upward.
2. **Lookahead bias in fundamentals** — yfinance `.info` returns current P/E, P/B, ROE, gross margin, market cap. These are used as static cross-sectional scores, not point-in-time historical values. Frame fundamental factors as a static ranking exercise, not a true historical simulation.
3. **Static market cap** — size factor uses current market cap for all months.
4. **Equal-weighted universe benchmark** — the benchmark is an equal-weighted average of all available stocks each month. This is NOT a cap-weighted SPY-like benchmark. Label it clearly as "equal-weighted universe benchmark" everywhere in the app and README.

These do not disqualify the project — they must be stated clearly. Institutional backtests use point-in-time databases (Compustat, Bloomberg). This project uses free data and is labelled accordingly.

---

## Architecture — 4 Agents, run strictly in order

---

### Agent 1: Data Ingestion
**File:** `agents/data_ingestion.py`

**Step 1 — Ticker list:**
- Pull S&P500 tickers from Wikipedia: `pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]`
- Extract the `Symbol` column
- **Normalise tickers to Yahoo Finance format:** replace `.` with `-` in all tickers (e.g. `BRK.B` → `BRK-B`, `BF.B` → `BF-B`). Without this step yfinance will silently fail to download some names.

**Step 2 — Price data:**
- Download daily adjusted close prices for all tickers using `yfinance.download(tickers, start='2023-01-01', end='2026-02-01', auto_adjust=True)`
- **Date range: 2023-01-01 through 2026-01-31** — reasons:
  - 2023 provides warm-up history for 12-month momentum calculations starting in January 2024
  - Prices through 2026-01-31 are required to compute forward returns for signals formed at 2025-12-31 (the final rebalance date needs a one-month-forward exit price)
- Download in batches of 50 tickers with `time.sleep(0.1)` between batches to respect rate limits
- Extract adjusted close prices only
- **Resample to business month-end:** use `resample('BME').last()` to get the last available trading day price for each calendar month. Use this month-end series consistently for all signal and return calculations. Never mix calendar month-end with trading month-end.
- Drop tickers with more than 20% missing monthly prices after resampling
- Save as `data/prices.parquet` — index is business month-end dates, columns are tickers

**Step 3 — Fundamental data:**
- For each ticker, fetch `.info` dict from yfinance
- Extract: `trailingPE`, `priceToBook`, `returnOnEquity`, `grossMargins`, `marketCap`
- Store raw values — do not drop rows with missing fields here, handle per-factor in Agent 2
- Save as `data/fundamentals.parquet` — one row per ticker

**Validation — print on completion:**
- Shape of prices DataFrame (rows = months, columns = tickers)
- Date range of prices index (min and max month-end date)
- Number of tickers successfully downloaded
- Number of tickers dropped for missing data
- Number of tickers with at least one fundamental field available
- Confirm prices index is business month-end (spot check first 5 index values)

---

### Agent 2: Factor Construction
**File:** `agents/factor_construction.py`

**Inputs:** `data/prices.parquet`, `data/fundamentals.parquet`

**Month-end calendar:**
- Signal formation dates: 2024-01-31 through 2025-12-31 (business month-end)
- These are the dates at which factor scores are computed and portfolios are formed
- The forward return for each formation date is computed in Agent 3 using the next month-end price
- The signal at 2025-12-31 is the last formation date; its forward return uses the 2026-01-31 price

**Factor construction — compute each factor independently per formation date:**

1. **Momentum (12–1 skip)**
   - At formation date `t` (a business month-end), compute:
     ```
     momentum[t] = price[t-1 month-end] / price[t-12 month-end] - 1
     ```
   - Explicitly: `t-1 month-end` is the previous business month-end; `t-12 month-end` is the business month-end 11 months before that (i.e. 12 months before `t`)
   - This excludes the most recent month's return (skip-1 convention) to avoid short-term reversal
   - Example: for formation date 2024-06-28, use price[2024-05-31] / price[2023-06-30] - 1
   - Requires at least 11 of 12 monthly prices to be non-NaN; otherwise NaN for that stock that month
   - Uses price data only — available for most tickers

2. **Value**
   - Score = negative of `trailingPE` (lower P/E = higher value score)
   - Fallback: if `trailingPE` missing, use negative of `priceToBook`
   - Static across all formation dates (acknowledged lookahead limitation)
   - If both missing: NaN

3. **Quality**
   - Score = `returnOnEquity`
   - Fallback: `grossMargins` if ROE missing
   - Static across all formation dates
   - If both missing: NaN

4. **Size**
   - Score = negative log of `marketCap` (smaller cap = higher score)
   - Static across all formation dates (acknowledged limitation)
   - If missing: NaN

5. **Low Volatility**
   - At formation date `t`, compute the standard deviation of daily returns over the 60 calendar days ending at `t`
   - Use raw (non-resampled) daily prices for this calculation
   - Score = negative of that 60-day daily return standard deviation (lower vol = higher score)
   - Requires at least 50 of 60 daily observations; otherwise NaN
   - Uses price data only — available for most tickers

**Z-scoring — critical:**
- For each factor, on each formation date, z-score cross-sectionally using only stocks with a valid (non-NaN) score for that factor
- Formula: `(x - mean(x)) / std(x)` where mean and std are computed from the available stocks only
- Do NOT do a global dropna across all five factors before z-scoring — each factor uses its own available universe independently

**Composite score:**
- For each stock each formation date, composite = mean of available factor z-scores
- Require at least 3 of 5 factor z-scores to be non-NaN for a stock to receive a composite score
- Stocks with fewer than 3 valid factors are excluded from the composite universe that month

**Output:**
- `data/factor_scores.parquet` — columns: [ticker, date, momentum, value, quality, size, low_vol, composite]
- One row per ticker per formation date (business month-end, 2024-01-31 through 2025-12-31)

**Validation — print on completion:**
- Number of stocks with valid score per factor per month (spot check 3 months)
- Cross-sectional mean and std of each z-scored factor (should be ~0 and ~1)
- Composite universe size per month (expect 400+ most months)
- Confirm no formation date uses prices beyond t-1 month-end for momentum (no lookahead in price-based factors)

---

### Agent 3: Backtesting
**File:** `agents/backtesting.py`

**Inputs:** `data/factor_scores.parquet`, `data/prices.parquet`

**Forward returns:**
- For each formation date `t`, the one-month forward return for stock `i` is:
  ```
  forward_return[i, t] = price[i, t+1 month-end] / price[i, t month-end] - 1
  ```
- Formation dates run 2024-01-31 through 2025-12-31
- The last formation date (2025-12-31) uses price[2026-01-31] as the exit — this is why prices extend to 2026-01-31
- Drop any stock-month where forward return cannot be computed (missing exit price)

**Portfolio construction — for each factor including composite:**
- At each formation date, rank all stocks with a valid score for that factor
- Assign to quintiles: Q1 = bottom 20% (lowest score), Q5 = top 20% (highest score)
- Equal-weight stocks within each quintile
- Portfolio return for that month = equal-weighted average of constituent forward returns

**Long-short spread:**
- Monthly spread return = Q5 return minus Q1 return for each factor

**Performance metrics — compute for each factor and composite:**
- Annualised return: `mean(monthly_spread_returns) * 12`
- Annualised volatility: `std(monthly_spread_returns) * sqrt(12)`
- Sharpe ratio: `(annualised_return - 0.05) / annualised_vol` (5% annual risk-free rate)
- Max drawdown: maximum peak-to-trough decline in cumulative Q5-Q1 return series
- Hit rate: % of months where spread return > 0

**Turnover — define precisely:**
- Compute **Q5 one-way turnover** for each factor: the fraction of the Q5 portfolio that changes between consecutive rebalances
  ```
  turnover[t] = |stocks entering Q5 at t| / |Q5 portfolio size at t|
  ```
- Report average Q5 one-way turnover across all months
- Also compute and report **combined long-short turnover** (average of Q5 and Q1 one-way turnovers) separately, labelled clearly
- Do not combine them into one number — report both

**Benchmark:**
- Equal-weighted universe benchmark = equal-weighted average monthly return across all stocks with a valid forward return that month
- Label as "equal-weighted universe benchmark" — NOT "S&P 500 benchmark"

**Outputs:**
- `data/backtest_results.parquet` — one row per factor, columns: [factor, ann_return, ann_vol, sharpe, max_drawdown, hit_rate, q5_one_way_turnover, ls_combined_turnover]
- `data/portfolio_returns.parquet` — monthly returns for Q1, Q3, Q5, long-short spread, and benchmark for each factor

**Validation — print on completion:**
- Full performance table for all 5 factors + composite (all metrics)
- First 10 rows of portfolio_returns.parquet
- Confirm last formation date (2025-12-31) has a valid forward return using 2026-01-31 price
- Confirm both parquet files written successfully

---

### Agent 4: Streamlit App
**File:** `app.py`

**Theme:** dark background, clean layout, Plotly charts throughout. Use `st.set_page_config(layout="wide")`.

**Sidebar navigation — 6 sections:**

**1. About**
- 2-paragraph plain-English explanation of what a factor model is and what this project does
- Methodology summary: 5 factors, monthly rebalance, equal-weight quintiles, skip-1 momentum, 2024–2025 evaluation period
- Disclaimer box (orange `st.warning`) stating explicitly:
  - Survivorship bias: uses today's S&P500 constituents
  - Fundamental lookahead: P/E, P/B, ROE, gross margin are current values not historical
  - Static market cap for size factor
  - Benchmark is equal-weighted universe, not cap-weighted SPY
  - Educational/portfolio project only — not investment advice

**2. Factor Performance**
- Horizontal bar chart: annualised Q5-Q1 return per factor, sorted descending
- Summary table: annualised return, Sharpe, max drawdown, hit rate, Q5 one-way turnover, L/S combined turnover — for all 5 factors + composite
- Colour-code Sharpe column: green > 0.5, yellow 0–0.5, red < 0

**3. Cumulative Returns**
- Line chart: cumulative Q5-Q1 return over 2024–2025 for all 5 factors + composite on same chart
- Horizontal dashed line at 0
- Plotly legend allows toggling individual factors on/off

**4. Quintile Breakdown**
- Dropdown to select a factor
- Grouped bar chart: annualised return for Q1, Q3, Q5 (skip Q2/Q4 for clarity)
- Caption explaining what monotonic Q1→Q5 spread means

**5. Stock Explorer**
- Dropdown: select any S&P500 ticker
- Table: factor z-score and quintile rank for each of the 5 factors as of most recent formation date
- Radar/spider chart of z-scores across the 5 factors
- Small metrics row: P/E, P/B, ROE, market cap from fundamentals

**6. Composite Portfolio**
- Line chart: cumulative return of composite Q5 portfolio vs equal-weighted universe benchmark
- Label benchmark clearly as "Equal-Weighted Universe Benchmark"
- Three metric cards side by side: annualised return, Sharpe, max drawdown — for composite Q5 vs benchmark

**Caching:**
- `@st.cache_data` on every data loading function
- Load from parquet files only — no yfinance calls at render time
- If any parquet file is missing, show `st.error("Data files not found. Run agents 1–3 first.")` and stop

---

## File Structure
```
sp500-factor-model/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── app.py
├── agents/
│   ├── data_ingestion.py
│   ├── factor_construction.py
│   └── backtesting.py
├── data/                        # commit to GitHub for Streamlit Cloud deployment
│   ├── prices.parquet
│   ├── fundamentals.parquet
│   ├── factor_scores.parquet
│   ├── backtest_results.parquet
│   └── portfolio_returns.parquet
├── utils/
│   └── helpers.py               # z_score(), sharpe(), max_drawdown(), quintile_assign()
└── .gitignore
```

**Deployment note:** Do NOT gitignore `data/`. Generate all parquet files locally by running agents 1–3, then commit `data/` to the repo. Streamlit Community Cloud cannot run the ingestion pipeline at deploy time.

**.gitignore:**
```
__pycache__/
*.pyc
.env
```

---

## requirements.txt
```
yfinance>=0.2.36
pandas>=2.0.0
numpy>=1.24.0
streamlit>=1.32.0
plotly>=5.18.0
pyarrow>=14.0.0
scikit-learn>=1.3.0
requests>=2.31.0
lxml>=4.9.0
```

---

## Antigravity Prompts — run strictly in order, one agent at a time

**Agent 1:**
Read CLAUDE.md. You are Agent 1: Data Ingestion. Build `agents/data_ingestion.py` exactly as specified. Key requirements: (1) normalise Wikipedia tickers to Yahoo format by replacing '.' with '-', (2) download prices from 2023-01-01 to 2026-02-01 so forward returns for the 2025-12-31 formation date can be computed, (3) resample adjusted close to business month-end using resample('BME').last() and use this consistently, (4) add time.sleep(0.1) between batches of 50 tickers. Run the script and print all validation stats. Do not build any other agent.

**Agent 2:**
Read CLAUDE.md. You are Agent 2: Factor Construction. Assume `data/prices.parquet` and `data/fundamentals.parquet` exist. Build `agents/factor_construction.py`. Key requirements: (1) momentum at formation date t = price[t-1 month-end] / price[t-12 month-end] - 1, explicitly excluding the current month (skip-1), (2) z-score each factor independently using only stocks with a valid score for that factor — do not dropna across all factors, (3) composite requires at least 3 of 5 valid factor z-scores, (4) formation dates are business month-ends from 2024-01-31 through 2025-12-31 only. Run and print all validation stats. Do not build any other agent.

**Agent 3:**
Read CLAUDE.md. You are Agent 3: Backtesting. Assume `data/factor_scores.parquet` and `data/prices.parquet` exist. Build `agents/backtesting.py`. Key requirements: (1) forward return for formation date t uses price at t+1 business month-end, (2) confirm the last formation date 2025-12-31 successfully computes a forward return using the 2026-01-31 price, (3) report Q5 one-way turnover and long-short combined turnover separately, (4) label benchmark as equal-weighted universe benchmark not S&P 500. Print the full performance table for all 5 factors + composite and first 10 rows of portfolio_returns.parquet. Confirm both files written. Do not build any other agent.

**Agent 4:**
Read CLAUDE.md. You are Agent 4: Streamlit App. Assume all parquet files in `data/` exist. Build `app.py` with all 6 sections as specified. Key requirements: (1) @st.cache_data on all data loading, (2) no yfinance calls at render time, (3) label benchmark as equal-weighted universe benchmark everywhere, (4) disclaimer box must mention all four limitations. Run `streamlit run app.py` and confirm it loads without errors. Do not modify any agent files.

---

## Deployment (Streamlit Community Cloud)
1. Run agents 1–3 locally to generate all parquet files in `data/`
2. Commit the `data/` folder to GitHub
3. Push repo to GitHub
4. Go to share.streamlit.io → New app → connect repo → main file: `app.py`
5. No secrets or environment variables needed
6. Add deployed URL to GitHub repo website field and top of README

---

## Citadel Interview Talking Points
- "I used skip-1 momentum — the signal at month-end t is computed from price[t-12] to price[t-1], explicitly excluding the most recent month to avoid short-term reversal contamination"
- "I z-score each factor cross-sectionally and independently each month using only stocks with a valid score for that factor, so sparse fundamental coverage doesn't distort the composite"
- "The composite requires at least 3 of 5 valid factor scores — this prevents stocks with thin coverage dominating the portfolio"
- "I extended prices through January 2026 so the final rebalance signal formed at December 2025 has a valid one-month-forward exit — without this the last data point is unusable"
- "The project has four documented limitations: survivorship bias, fundamental lookahead, static market cap, and an equal-weighted rather than cap-weighted benchmark. In production you'd use a point-in-time database like Compustat"
- "I report Q5 one-way turnover separately from long-short combined turnover — they tell different stories. Q5 one-way is relevant for a long-only manager; combined is relevant for a market-neutral fund"
- "Momentum has the highest turnover at roughly 30-40% monthly. At 5-10bps per side transaction cost that meaningfully erodes the gross spread — which is why many practitioners run momentum at lower frequency or apply turnover constraints"
- "The composite equal-weights all five factors. A more rigorous approach would optimise weights using a covariance-regularised information ratio, or orthogonalise factors using a factor risk model to remove overlap between, say, quality and low-vol"