import os
from dotenv import load_dotenv
import pandas as pd
from sklearn.metrics import accuracy_score
from src.fetch_data import fetch_history
from src.features import make_features
from src.model import train_and_save, load_model
from src.trade import close_position, get_position, place_bracket_buy

load_dotenv()


def main():
    ticker = os.getenv('TICKER', 'GC=F')
    interval = os.getenv('DATA_INTERVAL', '1m')
    period = os.getenv('DATA_PERIOD', '2d')
    print(f'Fetching {interval} data for {ticker}...')
    df = fetch_history(ticker=ticker, period=period, interval=interval)
    print('\n=== LATEST DATA (20 ROWS) ===')
    print(df.tail(20).to_string(index=False))
    print('Saved to data/latest_20.csv')
    X, y = make_features(df)
    if len(X) < 100:
        print('Not enough data to train. Need at least 100 rows of features.')
        return
    # Keep the newest row completely out of training so its prediction is unseen.
    samples = min(len(X), 500)
    X_recent = X.tail(samples)
    y_recent = y.tail(samples)
    validation_size = max(20, int(len(X_recent) * 0.2))
    if len(X_recent) <= validation_size:
        print('Not enough data for a separate validation set.')
        return
    X_train = X_recent.iloc[:-validation_size]
    y_train = y_recent.iloc[:-validation_size]
    X_validation = X_recent.iloc[-validation_size:-1]
    y_validation = y_recent.iloc[-validation_size:-1]
    clf = train_and_save(X_train, y_train)
    validation_predictions = clf.predict(X_validation)
    validation_win_rate = accuracy_score(y_validation, validation_predictions)
    predicted_buys = validation_predictions == 1
    if predicted_buys.any():
        buy_win_rate = accuracy_score(y_validation[predicted_buys], validation_predictions[predicted_buys])
    else:
        buy_win_rate = 0.0
    print('\n=== VALIDATION ===')
    print(f'Historical validation accuracy: {validation_win_rate:.1%} '
          f'({len(y_validation)} unseen predictions)')
    print(f'Predicted-buy win rate: {buy_win_rate:.1%} '
          f'({predicted_buys.sum()} historical buy signals)')
    model = load_model()
    latest = X.tail(1)
    probabilities = model.predict_proba(latest)[0]
    class_probabilities = dict(zip(model.classes_, probabilities))
    buy_probability = class_probabilities.get(1, 0.0)
    sell_probability = class_probabilities.get(0, 0.0)
    buy_threshold = float(os.getenv('BUY_THRESHOLD', '0.55'))
    sell_threshold = float(os.getenv('SELL_THRESHOLD', '0.55'))
    minimum_win_rate = float(os.getenv('MINIMUM_WIN_RATE', '0.52'))
    trading_enabled = os.getenv('ENABLE_TRADING', 'false').lower() == 'true'
    symbol = os.getenv('TRADE_SYMBOL', 'GLD')
    stop_loss_pct = float(os.getenv('STOP_LOSS_PCT', '0.01'))
    take_profit_pct = float(os.getenv('TAKE_PROFIT_PCT', '0.02'))
    latest_price = float(df['Close'].iloc[-1])
    action = 'HOLD'
    if buy_probability > buy_threshold and buy_win_rate >= minimum_win_rate:
        action = 'BUY' if trading_enabled else 'BUY BLOCKED'
    elif buy_probability > buy_threshold:
        action = 'BUY REJECTED'
    elif sell_probability > sell_threshold:
        action = 'SELL' if trading_enabled else 'SELL BLOCKED'
    prediction_table = pd.DataFrame([{
        'bar_time': df.index[-1],
        'data_timeframe': interval,
        'prediction_horizon': f'next {interval} bar',
        'data_ticker': ticker,
        'trade_symbol': symbol,
        'last_close': round(latest_price, 2),
        'buy_probability': f'{buy_probability:.1%}',
        'sell_probability': f'{sell_probability:.1%}',
        'buy_win_rate': f'{buy_win_rate:.1%}',
        'action': action,
        'stop_loss': round(latest_price * (1 - stop_loss_pct), 2),
        'take_profit': round(latest_price * (1 + take_profit_pct), 2),
    }])
    prediction_table.to_csv('data/latest_prediction.csv', index=False)
    print('\n=== TRADE PREDICTION TABLE ===')
    print('Saved to data/latest_prediction.csv')
    print(prediction_table.to_string(index=False))
    print('\n=== PREDICTIONS ===')
    print(f'Buy probability:  {buy_probability:.3f} (threshold: {buy_threshold:.3f})')
    print(f'Sell probability: {sell_probability:.3f} (threshold: {sell_threshold:.3f})')
    print(f'Minimum validation win rate: {minimum_win_rate:.1%}')
    print(f'Trading enabled: {trading_enabled}')
    if buy_probability > buy_threshold and buy_win_rate >= minimum_win_rate:
        qty = int(os.getenv('TRADE_QTY', '1'))
        print(f'Buy signal for {symbol} qty={qty}')
        if not trading_enabled:
            print('Order blocked: set ENABLE_TRADING=true after reviewing the signal.')
            return
        try:
            resp = place_bracket_buy(symbol, qty)
            print('Order response:', resp)
        except Exception as e:
            print('Order failed:', e)
    elif buy_probability > buy_threshold:
        print('Buy signal rejected: recent predicted-buy win rate is too low.')
    elif sell_probability > sell_threshold:
        print(f'Sell signal for {symbol}.')
        if not trading_enabled:
            print('Order blocked: set ENABLE_TRADING=true after reviewing the signal.')
            return
        try:
            if get_position(symbol) is None:
                print('No open position; no sell order placed.')
            else:
                print('Closing the existing position; attached bracket exits will be cancelled by Alpaca.')
                print('Close response:', close_position(symbol))
        except Exception as e:
            print('Close failed:', e)
    else:
        print('No buy or sell signal passed its threshold.')


if __name__ == '__main__':
    main()
