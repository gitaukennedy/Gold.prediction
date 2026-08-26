import os
from dotenv import load_dotenv
import pandas as pd
from sklearn.metrics import accuracy_score
from src.fetch_data import fetch_history
from src.features import make_features
from src.model import train_and_save, load_model
from src.trade import close_position, get_position, place_bracket_buy
from src.risk import build_risk_plan, markov_monte_carlo

load_dotenv()


def main():
    ticker = os.getenv('TICKER', 'GC=F')
    interval = os.getenv('DATA_INTERVAL', '1m')
    period = os.getenv('DATA_PERIOD', '2d')
    print(f'Fetching {interval} data for {ticker}...')
    df = fetch_history(ticker=ticker, period=period, interval=interval)
    latest_bar_time = pd.Timestamp(df.index[-1])
    if latest_bar_time.tzinfo is None:
        latest_bar_time = latest_bar_time.tz_localize('UTC')
    else:
        latest_bar_time = latest_bar_time.tz_convert('UTC')
    interval_unit = interval[-1].lower()
    interval_value = float(interval[:-1])
    interval_minutes = interval_value * {'m': 1, 'h': 60, 'd': 1440}[interval_unit]
    max_age_minutes = float(os.getenv(
        'MARKET_DATA_MAX_AGE_MINUTES',
        str(max(30, interval_minutes * 3)),
    ))
    data_age_minutes = (pd.Timestamp.now(tz='UTC') - latest_bar_time).total_seconds() / 60
    if data_age_minutes > max_age_minutes:
        print(f'Latest market bar is {data_age_minutes:.1f} minutes old; '
              'no current prediction or order created.')
        return
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
    latest_price = float(df['Close'].iloc[-1])
    risk_lookback = int(os.getenv('RISK_LOOKBACK_BARS', '20'))
    simulation_horizon = int(os.getenv('SIMULATION_HORIZON_BARS', '5'))
    monte_carlo = markov_monte_carlo(df, horizon=simulation_horizon)
    action = 'HOLD'
    if buy_probability > buy_threshold and buy_win_rate >= minimum_win_rate:
        action = 'BUY' if trading_enabled else 'BUY BLOCKED'
    elif buy_probability > buy_threshold:
        action = 'BUY REJECTED'
    elif sell_probability > sell_threshold:
        action = 'SELL' if trading_enabled else 'SELL BLOCKED'
    direction = 'CALL' if buy_probability >= sell_probability else 'PUT'
    risk_plan = build_risk_plan(df, direction, interval, risk_lookback)
    prediction_table = pd.DataFrame([{
        'run_time': pd.Timestamp.now(tz='UTC').isoformat(),
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
        'signal_direction': direction,
        'range_stop_loss': round(risk_plan['stop'], 2),
        'range_take_profit': round(risk_plan['target'], 2),
        'atr': round(risk_plan['atr'], 4),
        'monte_carlo_up_probability': f"{monte_carlo['up_probability']:.1%}",
        'monte_carlo_p05_return': f"{monte_carlo['p05_return']:.2%}",
        'monte_carlo_p95_return': f"{monte_carlo['p95_return']:.2%}",
    }])
    os.makedirs('data', exist_ok=True)
    history_path = 'data/prediction_history.csv'
    prediction_table.to_csv(
        history_path,
        mode='a',
        header=not os.path.exists(history_path),
        index=False,
    )
    print('\n=== TRADE PREDICTION TABLE ===')
    print('Appended prediction history to data/prediction_history.csv')
    print(prediction_table.to_string(index=False))
    print('\n=== PREDICTIONS ===')
    print(f'Buy probability:  {buy_probability:.3f} (threshold: {buy_threshold:.3f})')
    print(f'Sell probability: {sell_probability:.3f} (threshold: {sell_threshold:.3f})')
    print(f'Minimum validation win rate: {minimum_win_rate:.1%}')
    print(f"Markov Monte Carlo ({simulation_horizon} bars): up {monte_carlo['up_probability']:.1%}, "
          f"5-95% return {monte_carlo['p05_return']:.2%} to {monte_carlo['p95_return']:.2%}")
    print(f"{direction} range plan ({interval}; ATR {risk_plan['atr']:.4f}): "
          f"stop {risk_plan['stop']:.2f}, target {risk_plan['target']:.2f}")
    print(f'Trading enabled: {trading_enabled}')
    if buy_probability > buy_threshold and buy_win_rate >= minimum_win_rate:
        qty = int(os.getenv('TRADE_QTY', '1'))
        print(f'Buy signal for {symbol} qty={qty}')
        if not trading_enabled:
            print('Order blocked: set ENABLE_TRADING=true after reviewing the signal.')
            return
        try:
            resp = place_bracket_buy(symbol, qty, risk_plan['stop_distance_pct'],
                                     risk_plan['target_distance_pct'])
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
