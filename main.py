import os
from dotenv import load_dotenv
from src.fetch_data import fetch_history
from src.features import make_features
from src.model import train_and_save, load_model
from src.trade import place_order

load_dotenv()


def main():
    ticker = os.getenv('TICKER', 'GC=F')
    print(f'Fetching data for {ticker}...')
    df = fetch_history(ticker=ticker, period='2d', interval='1m')
    print('\n=== LATEST DATA (20 ROWS) ===')
    print(df.tail(20).to_string(index=False))
    print('Saved to data/latest_20.csv')
    X, y = make_features(df)
    if len(X) < 100:
        print('Not enough data to train. Need at least 100 rows of features.')
        return
    # Train on recent samples for speed
    clf = train_and_save(X.tail(500), y.tail(500))
    model = load_model()
    latest = X.tail(1)
    probabilities = model.predict_proba(latest)[0]
    class_probabilities = dict(zip(model.classes_, probabilities))
    buy_probability = class_probabilities.get(1, 0.0)
    sell_probability = class_probabilities.get(0, 0.0)
    buy_threshold = float(os.getenv('BUY_THRESHOLD', '0.55'))
    sell_threshold = float(os.getenv('SELL_THRESHOLD', '0.55'))
    print('\n=== PREDICTIONS ===')
    print(f'Buy probability:  {buy_probability:.3f} (threshold: {buy_threshold:.3f})')
    print(f'Sell probability: {sell_probability:.3f} (threshold: {sell_threshold:.3f})')
    if buy_probability > buy_threshold:
        qty = int(os.getenv('TRADE_QTY', '1'))
        symbol = os.getenv('TRADE_SYMBOL', 'GLD')
        print(f'Buy signal: probability is above {buy_threshold:.3f}; placing buy for {symbol} qty={qty} (paper)')
        try:
            resp = place_order(symbol, qty, side='buy')
            print('Order response:', resp)
        except Exception as e:
            print('Order failed:', e)
    elif sell_probability > sell_threshold:
        print('Sell signal: probability is above the sell threshold. No sell order placed.')
    else:
        print('No buy or sell signal passed its threshold.')


if __name__ == '__main__':
    main()
