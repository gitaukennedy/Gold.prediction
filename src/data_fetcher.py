"""Fetch live gold price data from yfinance"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoldDataFetcher:
    """Fetch gold futures data from Yahoo Finance"""
    
    def __init__(self, ticker: str = "GC=F", period: str = "3mo"):
        """
        Initialize Gold Data Fetcher
        
        Args:
            ticker: Gold futures ticker (GC=F for COMEX gold)
            period: Historical period (e.g., '3mo', '1y', '5y')
        """
        self.ticker = ticker
        self.period = period
        
    def fetch_historical_data(self) -> pd.DataFrame:
        """
        Fetch historical gold price data
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Fetching historical data for {self.ticker}...")
            data = yf.download(self.ticker, period=self.period, progress=False)
            
            if data is None or len(data) == 0:
                raise ValueError("No data retrieved")
            
            logger.info(f"Successfully fetched {len(data)} records")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise
    
    def fetch_latest_price(self) -> dict:
        """
        Fetch latest gold price
        
        Returns:
            Dictionary with latest price info
        """
        try:
            ticker = yf.Ticker(self.ticker)
            data = ticker.history(period="1d")
            
            if len(data) == 0:
                raise ValueError("No data available")
            
            latest = data.iloc[-1]
            return {
                "price": latest["Close"],
                "timestamp": data.index[-1],
                "high": latest["High"],
                "low": latest["Low"],
                "volume": latest["Volume"]
            }
        except Exception as e:
            logger.error(f"Error fetching latest price: {e}")
            raise


def main():
    """Test data fetcher"""
    fetcher = GoldDataFetcher(period="3mo")
    data = fetcher.fetch_historical_data()
    print(data.tail(10))
    print("\nLatest Price:", fetcher.fetch_latest_price())


if __name__ == "__main__":
    main()
