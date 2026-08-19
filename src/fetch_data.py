from typing import Optional
import os
import yfinance as yf
import pandas as pd


def fetch_history(ticker: str = "GC=F", period: str = "3d", interval: str = "1m") -> pd.DataFrame:
    """Fetch recent minute-level history for ticker using yfinance.

    Returns a DataFrame with Open/High/Low/Close/Volume and cleaned NA rows.
    """
    t = yf.Ticker(ticker)
    df = t.history(period=period, interval=interval)
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/latest.csv", index=False)
    df.tail(20).to_csv("data/latest_20.csv", index=False)
    return df
