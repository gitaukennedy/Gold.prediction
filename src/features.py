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
    df = df.dropna()
    feature_cols = [c for c in df.columns if c.startswith('lag_')] + ['ma_ratio']
    X = df[feature_cols]
    # target: 1 if next close > current close
    y = (df['Close'].shift(-1) > df['Close']).astype(int)
    X = X.iloc[:-1]
    y = y.iloc[:-1]
    return X, y
