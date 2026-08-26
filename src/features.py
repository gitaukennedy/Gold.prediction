import numpy as np
import pandas as pd


def make_features(df: pd.DataFrame, n_lags: int = 5):
    """Create simple lag and moving-average features. Returns X, y aligned so each row predicts next-close direction.
    """
    df = df.copy()
    df['return'] = df['Close'].pct_change()
    for i in range(1, n_lags + 1):
        df[f'lag_{i}'] = df['return'].shift(i)
    df['ma_5'] = df['Close'].rolling(5).mean()
    df['ma_15'] = df['Close'].rolling(15).mean()
    df['ma_ratio'] = df['ma_5'] / df['ma_15']
    df['trend_15'] = df['Close'] / df['ma_15'] - 1
    prior_close = df['Close'].shift(1)
    true_range = pd.concat([df['High'] - df['Low'], (df['High'] - prior_close).abs(),
                            (df['Low'] - prior_close).abs()], axis=1).max(axis=1)
    df['atr_pct'] = true_range.rolling(14).mean() / df['Close']
    rolling_low, rolling_high = df['Low'].rolling(20).min(), df['High'].rolling(20).max()
    df['range_position'] = (df['Close'] - rolling_low) / (rolling_high - rolling_low)
    df['realized_vol'] = df['return'].rolling(20).std()
    df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean().replace(0, np.nan)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    feature_cols = [c for c in df.columns if c.startswith('lag_')] + [
        'ma_ratio', 'trend_15', 'atr_pct', 'range_position', 'realized_vol', 'volume_ratio'
    ]
    X = df[feature_cols]
    # target: 1 if next close > current close
    y = (df['Close'].shift(-1) > df['Close']).astype(int)
    X = X.iloc[:-1]
    y = y.iloc[:-1]
    return X, y
