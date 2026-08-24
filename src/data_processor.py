"""Data preprocessing and technical indicator calculation"""
import pandas as pd
import numpy as np
import logging
from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoldDataProcessor:
    """Process gold price data and calculate technical indicators"""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            prices: Close prices series
            period: RSI period (default 14)
            
        Returns:
            RSI series
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            prices: Close prices series
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            
        Returns:
            MACD line and Signal line
        """
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        
        return macd, macd_signal
    
    @staticmethod
    def calculate_moving_averages(prices: pd.Series, short: int = 20, long: int = 50) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate simple moving averages
        
        Args:
            prices: Close prices series
            short: Short MA period
            long: Long MA period
            
        Returns:
            Short MA and Long MA
        """
        ma_short = prices.rolling(window=short).mean()
        ma_long = prices.rolling(window=long).mean()
        
        return ma_short, ma_long
    
    @staticmethod
    def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for ML model
        
        Args:
            data: Raw OHLCV data
            
        Returns:
            DataFrame with features
        """
        df = data.copy()
        
        # Technical indicators
        df['RSI'] = GoldDataProcessor.calculate_rsi(df['Close'])
        df['MACD'], df['MACD_Signal'] = GoldDataProcessor.calculate_macd(df['Close'])
        df['SMA_20'], df['SMA_50'] = GoldDataProcessor.calculate_moving_averages(df['Close'])
        
        # Price movements
        df['Daily_Return'] = df['Close'].pct_change()
        df['High_Low_Ratio'] = (df['High'] - df['Low']) / df['Close']
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        
        # Lag features
        for i in range(1, 4):
            df[f'Close_Lag_{i}'] = df['Close'].shift(i)
            df[f'Return_Lag_{i}'] = df['Daily_Return'].shift(i)
        
        # Target variable (predict next day's direction)
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        # Drop NaN values
        df = df.dropna()
        
        logger.info(f"Features prepared. Shape: {df.shape}")
        return df
    
    @staticmethod
    def split_train_test(data: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets
        
        Args:
            data: Processed data
            test_size: Test set fraction
            
        Returns:
            Train and test DataFrames
        """
        split_idx = int(len(data) * (1 - test_size))
        train = data[:split_idx]
        test = data[split_idx:]
        
        logger.info(f"Train: {len(train)}, Test: {len(test)}")
        return train, test


def main():
    """Test data processor"""
    from data_fetcher import GoldDataFetcher
    
    fetcher = GoldDataFetcher(period="3mo")
    raw_data = fetcher.fetch_historical_data()
    
    processed = GoldDataProcessor.prepare_features(raw_data)
    print(processed.tail(10))
    print("\nFeatures shape:", processed.shape)


if __name__ == "__main__":
    main()
