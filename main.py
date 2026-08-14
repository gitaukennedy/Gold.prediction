import os
from src.fetch_data import fetch_history
from src.features import make_features
from src.model import train_and_save, load_model
from src.trade import place_order


def main():
    ticker = os.getenv('TICKER', 'GC=F')
    print(f'Fetching data for {ticker}...')
    df = fetch_history(ticker=ticker, period='2d', interval='1m')
    X, y = make_features(df)
    if len(X) < 100:
        print('Not enough data to train. Need at least 100 rows of features.')
        return
    # Train on recent samples for speed
    clf = train_and_save(X.tail(500), y.tail(500))
    model = load_model()
    latest = X.tail(1)
    prob = model.predict_proba(latest)[0][1]
    print(f'Buy probability: {prob:.3f}')
    threshold = float(os.getenv('BUY_THRESHOLD', '0.55'))
    if prob > threshold:
        qty = int(os.getenv('TRADE_QTY', '1'))
        symbol = os.getenv('TRADE_SYMBOL', 'GLD')
        print(f'Prob {prob:.3f} > {threshold}, placing buy for {symbol} qty={qty} (paper)')
        try:
            resp = place_order(symbol, qty, side='buy')
            print('Order response:', resp)
        except Exception as e:
            print('Order failed:', e)
    else:
        print('No trade taken.')


if __name__ == '__main__':
    main()
