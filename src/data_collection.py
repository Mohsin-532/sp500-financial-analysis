import yfinance as yf
import pandas as pd
import datetime
import os

def collect_stock_data(tickers, start_date, end_date):
    """
    Collects historical stock data for the given tickers and date range.
    """
    print(f"Collecting historical stock data for: {', '.join(tickers)}")
    try:
        data = yf.download(tickers, start=start_date, end=end_date)
        # Flatten the MultiIndex columns returned by yf.download for multiple tickers
        data = data.stack(level=1).rename_axis(['Date', 'Ticker']).reset_index()
        return data
    except Exception as e:
        print(f"Error downloading stock data: {e}")
        return pd.DataFrame()

def collect_financial_data(tickers):
    """
    Collects key financial metrics (Market Cap, P/E Ratio, Revenue, Net Income) for the given tickers.
    """
    print("Collecting key financial metrics...")
    financial_data = []
    for ticker in tickers:
        print(f"Fetching financials for {ticker}...")
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Extract relevant metrics, handling missing data gracefully
            market_cap = info.get('marketCap', None)
            pe_ratio = info.get('trailingPE', None)
            revenue = info.get('totalRevenue', None)
            net_income = info.get('netIncomeToCommon', None)
            
            financial_data.append({
                'Ticker': ticker,
                'Market Cap': market_cap,
                'P/E Ratio': pe_ratio,
                'Revenue': revenue,
                'Net Income': net_income
            })
        except Exception as e:
             print(f"Error fetching financials for {ticker}: {e}")
             
    return pd.DataFrame(financial_data)

def main():
    # Define parameters
    tickers = [
        "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B", "TSLA", "UNH", "JNJ",
        "JPM", "V", "PG", "MA", "HD"
    ]
    start_date = "2015-01-01"
    end_date = "2025-01-01"
    
    # Define output paths
    raw_data_dir = os.path.join("data", "raw")
    os.makedirs(raw_data_dir, exist_ok=True)
    stock_data_file = os.path.join(raw_data_dir, "raw_stock_data.csv")
    financial_data_file = os.path.join(raw_data_dir, "raw_financial_data.csv")

    # Collect Data
    stock_df = collect_stock_data(tickers, start_date, end_date)
    financial_df = collect_financial_data(tickers)
    
    # Save to CSV
    if not stock_df.empty:
        stock_df.to_csv(stock_data_file, index=False)
        print(f"Saved stock data to {stock_data_file} ({len(stock_df)} rows)")
    else:
         print("Warning: Stock data is empty.")
         
    if not financial_df.empty:
        financial_df.to_csv(financial_data_file, index=False)
        print(f"Saved financial data to {financial_data_file} ({len(financial_df)} rows)")
    else:
        print("Warning: Financial data is empty.")

if __name__ == "__main__":
    main()
