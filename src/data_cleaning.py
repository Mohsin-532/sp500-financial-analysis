import pandas as pd
import numpy as np
import os


# ─────────────────────────────────────────────
# 1. HANDLE MISSING VALUES
# ─────────────────────────────────────────────
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing values in the DataFrame.
    - Numeric columns: filled forward then backward (time-series safe), then zeroed.
    - String columns: filled with 'Unknown'.
    Returns the cleaned DataFrame.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    other_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # For numeric: forward-fill first (carry last known value), then backward-fill
    df[numeric_cols] = df[numeric_cols].ffill().bfill().fillna(0)

    # For non-numeric (except Date/Ticker which should already be set)
    for col in other_cols:
        if col not in ['Date', 'Ticker']:
            df[col] = df[col].fillna('Unknown')

    print(f"  ✔ Missing values handled. Remaining NaNs: {df.isnull().sum().sum()}")
    return df


# ─────────────────────────────────────────────
# 2. STANDARDIZE DATES
# ─────────────────────────────────────────────
def standardize_dates(df: pd.DataFrame, date_col: str = 'Date') -> pd.DataFrame:
    """
    Converts the date column to a proper datetime object and sorts by it.
    """
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.sort_values(date_col).reset_index(drop=True)
        print(f"  ✔ Dates standardized. Range: {df[date_col].min().date()} → {df[date_col].max().date()}")
    else:
        print(f"  ⚠ Column '{date_col}' not found. Skipping date standardization.")
    return df


# ─────────────────────────────────────────────
# 3. CALCULATE FINANCIAL METRICS
# ─────────────────────────────────────────────
def calculate_volatility(df: pd.DataFrame, price_col: str = 'Close') -> pd.DataFrame:
    """
    Calculates annualised volatility for each ticker using the standard deviation
    of daily log returns (×√252 trading days).
    Adds a 'Volatility' column to the DataFrame.
    """
    df = df.copy()
    df['Daily_Return'] = df.groupby('Ticker')[price_col].transform(
        lambda x: np.log(x / x.shift(1))
    )
    df['Volatility'] = df.groupby('Ticker')['Daily_Return'].transform(
        lambda x: x.std() * np.sqrt(252)
    )
    df.drop(columns=['Daily_Return'], inplace=True)
    print("  ✔ Annualised volatility calculated.")
    return df


def calculate_revenue_growth(financial_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Year-over-Year revenue growth (%) from the financial data.
    Revenue growth = (Current Revenue - Previous Revenue) / |Previous Revenue| × 100
    Adds a 'Revenue_Growth_%' column.
    """
    financial_df = financial_df.copy()
    if 'Revenue' in financial_df.columns:
        financial_df['Revenue_Growth_%'] = financial_df.groupby('Ticker')['Revenue'].transform(
            lambda x: x.pct_change() * 100
        ).round(2)
        print("  ✔ Revenue growth (YoY %) calculated.")
    else:
        print("  ⚠ 'Revenue' column not found. Skipping revenue growth.")
    return financial_df


def calculate_pe_ratio(financial_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures P/E ratio exists and is numeric. If missing, attempts to derive it
    from Market Cap and Net Income. Adds/validates the 'P/E_Ratio' column.
    """
    financial_df = financial_df.copy()

    if 'P/E Ratio' not in financial_df.columns:
        # Attempt to derive: P/E ≈ Market Cap / Net Income (proxy)
        if 'Market Cap' in financial_df.columns and 'Net Income' in financial_df.columns:
            financial_df['P/E Ratio'] = (
                financial_df['Market Cap'] / financial_df['Net Income'].replace(0, np.nan)
            ).round(2)
            print("  ✔ P/E Ratio derived from Market Cap / Net Income.")
        else:
            print("  ⚠ Cannot calculate P/E Ratio — missing source columns.")
    else:
        financial_df['P/E Ratio'] = pd.to_numeric(financial_df['P/E Ratio'], errors='coerce').round(2)
        print("  ✔ P/E Ratio validated.")
    return financial_df


def rank_by_market_cap(financial_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'Market_Cap_Rank' column where 1 = largest market cap.
    """
    financial_df = financial_df.copy()
    if 'Market Cap' in financial_df.columns:
        financial_df['Market_Cap_Rank'] = financial_df['Market Cap'].rank(
            ascending=False, method='min', na_option='bottom'
        ).astype(int)
        print("  ✔ Market cap rankings assigned.")
    else:
        print("  ⚠ 'Market Cap' column not found. Skipping ranking.")
    return financial_df


# ─────────────────────────────────────────────
# 4. MAIN PIPELINE
# ─────────────────────────────────────────────
def main():
    raw_dir = os.path.join("data", "raw")
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    stock_raw_path = os.path.join(raw_dir, "raw_stock_data.csv")
    financial_raw_path = os.path.join(raw_dir, "raw_financial_data.csv")
    stock_out_path = os.path.join(processed_dir, "cleaned_stock_data.csv")
    financial_out_path = os.path.join(processed_dir, "cleaned_financial_data.csv")

    # ── Stock Data ──────────────────────────────
    print("\n[1/2] Cleaning stock data...")
    stock_df = pd.read_csv(stock_raw_path)
    stock_df = standardize_dates(stock_df, date_col='Date')
    stock_df = handle_missing_values(stock_df)
    stock_df = calculate_volatility(stock_df, price_col='Close')
    stock_df.to_csv(stock_out_path, index=False)
    print(f"  ✔ Saved cleaned stock data → {stock_out_path} ({len(stock_df)} rows)\n")

    # ── Financial Data ──────────────────────────
    print("[2/2] Cleaning financial data...")
    fin_df = pd.read_csv(financial_raw_path)
    fin_df = handle_missing_values(fin_df)
    fin_df = calculate_revenue_growth(fin_df)
    fin_df = calculate_pe_ratio(fin_df)
    fin_df = rank_by_market_cap(fin_df)
    fin_df.to_csv(financial_out_path, index=False)
    print(f"  ✔ Saved cleaned financial data → {financial_out_path} ({len(fin_df)} rows)\n")

    print("✅ Data cleaning complete!")


if __name__ == "__main__":
    main()
