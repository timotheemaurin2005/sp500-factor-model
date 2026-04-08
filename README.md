# S&P500 Multi-Factor Model

**[🚀 Live App](https://sp500-factor-model-tm.streamlit.app)** · Python · Pandas · Streamlit · Plotly

A fully deployed multi-factor equity model covering 498 S&P500 constituents over 2024–2025. Constructs, backtests, and visualises five systematic factors: Momentum, Value, Quality, Size, and Low Volatility, using a clean monthly rebalancing framework with equal-weighted quintile portfolios.

---

## What It Does

Factor models decompose stock returns into systematic drivers rather than treating each stock independently. This project implements the full pipeline:

1. **Data Ingestion** — downloads 3 years of adjusted price history and current fundamentals for ~500 tickers via yfinance, resampled to business month-end
2. **Factor Construction** — computes five cross-sectionally z-scored factor signals each month
3. **Backtesting** — forms equal-weighted quintile portfolios monthly, measures Q5–Q1 long-short spread performance
4. **Visualisation** — interactive Streamlit app with cumulative returns, quintile breakdowns, a stock explorer, and composite portfolio analysis

---

## Factors

| Factor | Construction | Signal |
|---|---|---|
| **Momentum** | Price return from t−12 to t−1 month-end (skip-1) | Higher past return → higher score |
| **Value** | Negative trailing P/E (fallback: P/B) | Lower valuation multiple → higher score |
| **Quality** | Return on Equity (fallback: gross margin) | Higher profitability → higher score |
| **Size** | Negative log market cap | Smaller company → higher score |
| **Low Volatility** | Negative 60-day daily return std dev | Lower volatility → higher score |

All factors are cross-sectionally z-scored independently each month. The **Composite** score is the equal-weighted mean of available factor z-scores, requiring at least 3 of 5 valid scores for inclusion.

---

## Results (2024–2025)

| Factor | Ann. Return | Sharpe | Max DD | Hit Rate |
|---|---|---|---|---|
| Momentum | +16.45% | +0.83 | -9.03% | 58.3% |
| Quality | +13.75% | +1.17 | -4.64% | 58.3% |
| Value | -20.20% | -2.16 | -33.32% | 29.2% |
| Low Volatility | -22.64% | -1.54 | -40.76% | 25.0% |
| Size | -26.58% | -3.88 | -41.22% | 8.3% |
| **Composite** | -20.48% | -1.89 | -37.29% | 33.3% |

**Equal-weighted universe benchmark:** ~17.95% annualised over the same period.

Momentum and Quality were the only two factors with positive long-short spread in this period — consistent with a large-cap growth dominated market (2024–2025) that penalised cheap small caps and defensive low-volatility names.

---

## App Sections

- **About** — methodology overview and documented limitations
- **Factor Performance** — annualised returns, Sharpe ratios, drawdowns, turnover
- **Cumulative Returns** — interactive line chart, toggle factors on/off
- **Quintile Breakdown** — Q1/Q3/Q5 returns per factor
- **Stock Explorer** — per-stock factor scores, quintile ranks, radar chart, fundamentals
- **Composite Portfolio** — composite Q5 vs equal-weighted universe benchmark

---

## Known Limitations

This is an educational/portfolio project using free public data. Three biases are explicitly acknowledged:

- **Survivorship bias** — uses today's S&P500 constituents; delisted stocks from 2024–2025 are excluded
- **Fundamental lookahead** — P/E, P/B, ROE, and gross margin are current values from yfinance, not point-in-time historical data. Fundamental factors (Value, Quality, Size) should be interpreted as static cross-sectional rankings, not true historical simulations
- **Static market cap** — Size factor uses current market cap for all months

In production, point-in-time databases (Compustat, Bloomberg) would be used to eliminate these biases.

---

## Stack

`Python` `pandas` `numpy` `yfinance` `Streamlit` `Plotly` `pyarrow`

---

## Run Locally

```bash
git clone https://github.com/timotheemaurin2005/sp500-factor-model
cd sp500-factor-model
pip install -r requirements.txt

# Data is pre-generated in data/ — run the app directly
streamlit run app.py

# To regenerate data from scratch:
python agents/data_ingestion.py
python agents/factor_construction.py
python agents/backtesting.py
streamlit run app.py
```

---

## Project Structure

```
sp500-factor-model/
├── app.py                      # Streamlit frontend
├── agents/
│   ├── data_ingestion.py       # Price + fundamental data pipeline
│   ├── factor_construction.py  # Factor scoring + z-scoring
│   └── backtesting.py          # Quintile portfolios + performance metrics
├── data/                       # Pre-generated parquet files
├── utils/
│   └── helpers.py
└── requirements.txt
```
