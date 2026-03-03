# S&P 500 Financial Analysis — Project Report

---

## Section 1: Data Collection

### 1.1 Data Source

Financial and stock market data were collected from **Yahoo Finance** using the `yfinance` Python library (v0.2+). Yahoo Finance was selected as the primary data source for the following reasons:

- **Free and open access** — no API key required for standard use
- **Comprehensive coverage** — provides both historical price data and key financial fundamentals
- **Python integration** — the `yfinance` library allows structured, reproducible data retrieval
- **Reliability** — Yahoo Finance is a widely cited source in academic and financial research

### 1.2 Companies Selected

We analysed **15 major S&P 500 companies** spanning multiple sectors, selected based on highest market capitalisation and sector diversity:

| Ticker | Company                  | Sector         |
|--------|--------------------------|----------------|
| AAPL   | Apple Inc.               | Technology     |
| MSFT   | Microsoft Corporation    | Technology     |
| NVDA   | NVIDIA Corporation        | Technology     |
| GOOGL  | Alphabet Inc.            | Communication  |
| META   | Meta Platforms Inc.      | Communication  |
| AMZN   | Amazon.com Inc.          | Consumer Disc. |
| TSLA   | Tesla Inc.               | Consumer Disc. |
| BRK-B  | Berkshire Hathaway       | Financials     |
| JPM    | JPMorgan Chase & Co.     | Financials     |
| V      | Visa Inc.                | Financials     |
| MA     | Mastercard Inc.          | Financials     |
| UNH    | UnitedHealth Group       | Healthcare     |
| JNJ    | Johnson & Johnson        | Healthcare     |
| PG     | Procter & Gamble Co.     | Consumer Stapl.|
| HD     | The Home Depot Inc.      | Consumer Disc. |

These companies were chosen because they represent the largest and most influential firms in the U.S. economy, offer coverage across diverse sectors, and have consistent data availability over the full 10-year study period.

### 1.3 Time Period

The study covers stock performance and financial data from **1st January 2015 to 1st January 2025** — a 10-year window. This period was selected to:

- Capture long-term growth trends (e.g., FAANG dominance)
- Include major economic events: COVID-19 crash (2020), tech bull run (2020–2021), rate hike cycle (2022–2023)
- Provide sufficient data for statistically meaningful volatility calculations

### 1.4 Variables Collected

#### Historical Stock Data (`raw_stock_data.csv`)
| Variable      | Description                               |
|---------------|-------------------------------------------|
| Date          | Trading date                              |
| Open          | Opening price (USD)                       |
| High          | Daily high price (USD)                    |
| Low           | Daily low price (USD)                     |
| Close         | Closing price (USD)                       |
| Adj Close     | Adjusted closing price (splits, dividends)|
| Volume        | Number of shares traded                   |
| Ticker        | Company stock symbol                      |

#### Financial Fundamentals (`raw_financial_data.csv`)
| Variable    | Description                               |
|-------------|-------------------------------------------|
| Ticker      | Company stock symbol                      |
| Market Cap  | Total market capitalisation (USD)         |
| P/E Ratio   | Price-to-Earnings ratio (trailing)        |
| Revenue     | Total annual revenue (USD)                |
| Net Income  | Net income attributable to shareholders   |

### 1.5 Code Snippet

```python
import yfinance as yf
import pandas as pd

tickers = ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL",
           "META", "BRK-B", "TSLA", "UNH", "JNJ",
           "JPM", "V", "PG", "MA", "HD"]

# Download 10 years of daily stock data
data = yf.download(tickers, start="2015-01-01", end="2025-01-01")
data = data.stack(level=1).rename_axis(['Date', 'Ticker']).reset_index()
data.to_csv("data/raw/raw_stock_data.csv", index=False)
```

---

## Section 2: Data Cleaning & Preparation

### 2.1 Overview

The raw data was cleaned and enriched using modular, reusable Python functions defined in `src/data_cleaning.py`. Each function has a single responsibility, making the pipeline transparent and reproducible.

### 2.2 Handling Missing Values

```python
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Forward-fill then backward-fill to preserve time-series continuity
    df[numeric_cols] = df[numeric_cols].ffill().bfill().fillna(0)
    return df
```

**Approach:** Numeric columns are forward-filled first (carrying the last valid observation forward, appropriate for time-series market data), then backward-filled for any leading NaNs. This avoids introducing artificial values via simple mean/median imputation, which would distort trend analysis.

### 2.3 Date Standardisation

```python
def standardize_dates(df: pd.DataFrame, date_col: str = 'Date') -> pd.DataFrame:
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.sort_values(date_col).reset_index(drop=True)
    return df
```

All date fields are parsed into `datetime64` objects and the DataFrame is sorted chronologically.

### 2.4 Financial Metrics Calculated

#### Annualised Volatility
Volatility measures the degree of price fluctuation and is a standard risk metric:

```python
def calculate_volatility(df, price_col='Close'):
    df['Daily_Return'] = df.groupby('Ticker')[price_col].transform(
        lambda x: np.log(x / x.shift(1))
    )
    df['Volatility'] = df.groupby('Ticker')['Daily_Return'].transform(
        lambda x: x.std() * np.sqrt(252)  # annualised (252 trading days)
    )
    return df
```

#### Revenue Growth (Year-over-Year %)
```python
def calculate_revenue_growth(df):
    df['Revenue_Growth_%'] = df.groupby('Ticker')['Revenue'].transform(
        lambda x: x.pct_change() * 100
    ).round(2)
    return df
```

#### P/E Ratio Validation
The P/E Ratio (Price-to-Earnings) sourced from Yahoo Finance is validated and coerced to numeric. If absent, it is derived as `Market Cap / Net Income`.

#### Market Cap Ranking
```python
def rank_by_market_cap(df):
    df['Market_Cap_Rank'] = df['Market Cap'].rank(ascending=False, method='min').astype(int)
    return df
```
Ranks companies 1–15, where 1 = largest market cap.

### 2.5 Output Files

| File | Location | Description |
|------|----------|-------------|
| `cleaned_stock_data.csv` | `data/processed/` | Daily OHLCV + Volatility, 2015–2025 |
| `cleaned_financial_data.csv` | `data/processed/` | Fundamentals + Revenue Growth, P/E, Market Cap Rank |

### 2.6 Summary of Changes Applied

| Issue | Treatment |
|-------|-----------|
| Missing numeric values | Forward-fill → back-fill → zero-fill |
| Date format inconsistency | Parsed to `datetime64`, sorted ascending |
| Volatility (not in raw data) | Calculated from log returns × √252 |
| Revenue Growth (not in raw data) | Calculated YoY percentage change |
| Market Cap Ranking (not in raw data) | Ranked 1–15 by descending market cap |
