from typing import Optional
import os
import yfinance as yf
import pandas as pd


def ticker_filename(ticker: str) -> str:
    """Return a filesystem-safe, stable name for a market-data ticker."""
    return ''.join(character if character.isalnum() else '_' for character in ticker)


def fetch_history(ticker: str = "GC=F", period: str = "3d", interval: str = "1m") -> pd.DataFrame:
    """Fetch recent minute-level history for ticker using yfinance.

    Returns a DataFrame with Open/High/Low/Close/Volume and cleaned NA rows.
    """
    os.makedirs("data", exist_ok=True)
    # The default yfinance cache is in the user profile, which is not always
    # writable (for example, in a sandboxed scheduled run).
    yf.set_tz_cache_location(os.path.abspath("data/.yfinance"))
    t = yf.Ticker(ticker)
    df = t.history(period=period, interval=interval)
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    name = ticker_filename(ticker)
    # Keep one raw-data snapshot per instrument so an XAUUSD prediction does
    # not replace the COMEX data used by the existing GLD workflow.
    df.to_csv(f"data/latest_{name}.csv", index=False)
    df.tail(20).to_csv(f"data/latest_20_{name}.csv", index=False)
    return df
