"""Timeframe-aware range exits and Markov/Monte Carlo market context."""
import numpy as np
import pandas as pd


def build_risk_plan(df: pd.DataFrame, side: str, interval: str, lookback: int = 20):
    """Use ATR plus recent support/resistance, rather than fixed percentages."""
    if len(df) < lookback + 14:
        raise ValueError('Not enough candles to create a range-based risk plan.')
    bars = df.iloc[-max(lookback, 14):]
    prior = df['Close'].shift(1)
    tr = pd.concat([df['High'] - df['Low'], (df['High'] - prior).abs(),
                    (df['Low'] - prior).abs()], axis=1).max(axis=1)
    entry, atr = float(df['Close'].iloc[-1]), float(tr.rolling(14).mean().iloc[-1])
    support, resistance = float(bars['Low'].min()), float(bars['High'].max())
    side = side.upper()
    if side == 'CALL':
        stop, target = min(entry - 1.2 * atr, support - .15 * atr), max(entry + 1.8 * atr, resistance)
    elif side == 'PUT':
        stop, target = max(entry + 1.2 * atr, resistance + .15 * atr), min(entry - 1.8 * atr, support)
    else:
        raise ValueError("side must be 'CALL' or 'PUT'")
    return {'side': side, 'timeframe': interval, 'entry': entry, 'atr': atr,
            'support': support, 'resistance': resistance, 'stop': stop, 'target': target,
            'stop_distance_pct': abs(entry - stop) / entry,
            'target_distance_pct': abs(target - entry) / entry}


def markov_monte_carlo(df: pd.DataFrame, paths: int = 3000, horizon: int = 5, seed: int = 42):
    """Estimate a 3-state return Markov chain and simulate the next N bars."""
    returns = df['Close'].pct_change().dropna().tail(300)
    volatility = returns.std()
    neutral = {'up_probability': .5, 'median_return': .0, 'p05_return': .0, 'p95_return': .0}
    if len(returns) < 30 or not np.isfinite(volatility) or volatility == 0:
        return neutral
    states = np.select([returns < -.25 * volatility, returns > .25 * volatility], [0, 2], default=1)
    transition = np.ones((3, 3))  # Laplace smoothing keeps sparse states usable.
    for old, new in zip(states[:-1], states[1:]):
        transition[old, new] += 1
    transition /= transition.sum(axis=1, keepdims=True)
    pools = [returns.iloc[states == state].to_numpy() for state in range(3)]
    rng, results, current = np.random.default_rng(seed), np.zeros(paths), int(states[-1])
    for path in range(paths):
        state, total = current, .0
        for _ in range(horizon):
            state = rng.choice(3, p=transition[state])
            total += rng.choice(pools[state] if len(pools[state]) else returns.to_numpy())
        results[path] = total
    return {'up_probability': float((results > 0).mean()), 'median_return': float(np.median(results)),
            'p05_return': float(np.quantile(results, .05)), 'p95_return': float(np.quantile(results, .95))}
