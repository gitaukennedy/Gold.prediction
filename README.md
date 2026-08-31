# Gold prediction and Alpaca trading MVP
REACH OUT  TO ME AT MY EMAIL FOR MORE QUANTITATIVE LIVE ANALYSIS
This project fetches recent gold-market data, creates causal price/range/volatility features, and trains a leakage-aware Random Forest-to-XGBoost stacked classifier. It also reports a Markov/Monte Carlo range scenario and uses the current candle timeframe's ATR and recent high/low range to create exits.


# It is a proof of concept, not a validated trading strategy. The model does not account for spread, slippage, commissions, market hours, position size, portfolio exposure, stop losses, or drawdown. Use paper trading first and review every signal yourself. naitengeneza bado , 

This repository contains a 30-minute MVP for live gold prediction and trading. It fetches minute-level gold data from yfinance, trains a lightweight RandomForest to predict short-term direction, and (optionally) places trades via Alpaca (paper/live) when the model signals a buy. "am still working on the sell signal

## Project flow

```text
yfinance (GC=F) -> data/latest.csv
			  |
			  v
		  feature creation
			  |
			  v
	RandomForest out-of-fold probability -> XGBoost train/save/load
			  |
			  v
	    buy/sell probability report
			  |
	    buy signal only -> Alpaca
```

The default market-data ticker is `GC=F` (COMEX gold futures). The default order symbol is `GLD` (the SPDR Gold Shares ETF); these are different instruments and are not interchangeable. `XAUUSD=X` can be added through `ADDITIONAL_PREDICTION_TICKERS` to generate an independent spot-gold forecast with the same model settings. Additional tickers are prediction-only and never submit an Alpaca order. Yahoo Finance must supply the ticker's intraday OHLCV data; if it does not, the program reports the unavailable feed and creates no forecast rather than substituting COMEX futures for spot gold.

## Repository files

### Application code

- `main.py`: Application orchestrator. Loads `.env`, fetches data, trains on the latest 500 usable rows, reports stacked-model probabilities plus risk/simulation context, and only submits a buy when the configured checks pass.
- `src/__init__.py`: Package marker for the `src` Python package.
- `src/fetch_data.py`: Calls `yfinance.Ticker.history()`, keeps `Open`, `High`, `Low`, `Close`, and `Volume`, removes missing rows, and writes separate `data/latest_<ticker>.csv` and `data/latest_20_<ticker>.csv` snapshots.
- `src/features.py`: Computes lagged returns, trend, ATR percentage, range position, realised volatility, and volume ratio. The label is `1` when the next close is higher than the current close, otherwise `0`.
- `src/model.py`: Fits a Random Forest, creates chronological out-of-fold Random Forest probabilities, and uses those as an additive feature for XGBoost. The saved model is `models/stacked_rf_xgb.pkl`.
- `src/risk.py`: Makes timeframe-aware CALL/PUT risk plans from a 14-bar ATR and the recent support/resistance range, and produces a three-state Markov Monte Carlo scenario.
- `src/trade.py`: Builds an Alpaca `TradingClient` from environment credentials and submits market orders. The base URL determines paper versus live mode; the default is Alpaca paper trading.

### Configuration and operations

- `config.example.env`: Safe configuration template. Copy it to `.env`; never commit real keys.
- `requirements.txt`: Pinned or bounded Python dependencies: yfinance, pandas, scikit-learn, joblib, alpaca-py, and python-dotenv.
- `scripts/run_mvp.bat`: Windows helper that installs dependencies and runs `main.py`. Git Bash users can run the equivalent commands shown below.
- `scripts/run_mvp.sh`: Git Bash helper that creates/uses `.venv`, installs requirements, and runs the predictor.
- `.gitignore`: Excludes secrets, virtual environments, Python cache files, and generated model files from Git.
- `commitmsg.txt`: Text used as a commit-message note; it is not needed to run the application.

### Generated or supporting files

- `data/latest.csv`: Latest downloaded history. Replaced on each successful run and ignored only if local Git rules are extended to ignore it.
- `data/latest_20.csv`: Last 20 downloaded rows for quick inspection.
- `data/prediction_history.csv`: Append-only prediction history. Every run adds one row with the UTC run time and prediction details.
- `models/README.md`: Explains that trained model binaries are generated at runtime.
- `models/rf.pkl`: Generated Random Forest model. It is ignored by Git and recreated on each run.
- `.env`: Local secrets and settings copied from `config.example.env`; ignored by Git and intentionally not documented with its values.
- `.venv/`: Local Python virtual environment; ignored by Git.
- `__pycache__/` and `*.pyc`: Python runtime cache files; ignored by Git.

## Git Bash setup

From the project directory:

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
cp config.example.env .env
```

Edit `.env` with Alpaca credentials. For paper trading, keep:

```dotenv
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

Run the predictor with:

```bash
python main.py
```

Or use the Git Bash helper, which creates the virtual environment and installs
the dependencies before running it:

```bash
bash scripts/run_mvp.sh
```

If XGBoost needs to be installed separately, activate the environment first,
then run:

```bash
source .venv/Scripts/activate
python -m pip install "xgboost>=2.1,<3"
```

The script needs network access to Yahoo Finance. It needs valid Alpaca paper credentials only when a buy signal passes the threshold, because that is when `place_order()` is called.

## Model, simulation, and exit logic

The model is a chronological stacked ensemble. A Random Forest first produces
out-of-fold up probabilities; XGBoost receives those probabilities as one
additional feature alongside returns, trend, ATR, volatility, range position,
and volume features. Out-of-fold construction is important: the second-stage
model does not receive a Random Forest prediction trained on the same target.
XGBoost is the preferred second-stage model. If its optional Windows wheel has
not yet installed, the same pipeline uses scikit-learn's histogram gradient
boosting as a clearly local fallback; installing XGBoost switches back to it
automatically.

Each run also estimates a three-state (down/neutral/up) Markov transition
matrix from recent returns and samples 3,000 Monte Carlo paths over
`SIMULATION_HORIZON_BARS`. Its probability and 5th/95th percentile returns are
context only; they do not guarantee a trade outcome.

For a CALL signal, the stop is below both an ATR buffer and recent support, and
the target is at least an ATR projection or recent resistance. A PUT plan uses
the inverse levels. When `TICKER` and `TRADE_SYMBOL` differ (for example
`GC=F` and `GLD`), the program transfers only the stop/target *percentage
distance* to the live quote. It never sends a futures price as an ETF exit.

## Paper-trading authorization

Keep credentials in a local, git-ignored `.env` file in this project folder.
For a paper-only order, the configuration must include:

```dotenv
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ENABLE_TRADING=true
TRADE_SYMBOL=GLD
TRADE_QTY=1
```

Do not put a credential file in Git or paste its values into chat. The script
will still refuse a duplicate long position and will print the risk plan before
submitting a qualifying order.

## Configuration reference

| Variable | Default | Meaning |
| --- | --- | --- |
| `ALPACA_API_KEY` / `APCA_API_KEY_ID` | none | Alpaca API key |
| `ALPACA_SECRET_KEY` / `APCA_API_SECRET_KEY` | none | Alpaca secret |
| `ALPACA_BASE_URL` / `APCA_API_BASE_URL` | paper URL | Paper or live Alpaca endpoint |
| `TICKER` | `GC=F` | yfinance data ticker |
| `ADDITIONAL_PREDICTION_TICKERS` | `XAUUSD=X` | Optional comma-separated Yahoo Finance tickers forecast with the same settings; prediction-only, never traded |
| `DATA_INTERVAL` | `1m` | Candle timeframe used for features and prediction; current default is one minute |
| `DATA_PERIOD` | `2d` | History downloaded for training |
| `MARKET_DATA_MAX_AGE_MINUTES` | `30` | Maximum age of the newest downloaded bar before prediction and trading are skipped |
| `TRADE_SYMBOL` | `GLD` | Alpaca order symbol |
| `BUY_THRESHOLD` | `0.55` | Buy probability must be greater than this value |
| `SELL_THRESHOLD` | `0.55` | Sell probability threshold; currently report-only |
| `MINIMUM_WIN_RATE` | `0.52` | Minimum recent predicted-buy win rate required before a buy |
| `TRADE_QTY` | `1` | Whole-unit quantity for the buy order |
| `TRADE_SYMBOL` | `GLD` | Alpaca order symbol |
| `ENABLE_TRADING` | `false` | Must be `true` before any order can be submitted |
| `STOP_LOSS_PCT` | `0.01` | Bracket stop loss below the estimated entry price |
| `TAKE_PROFIT_PCT` | `0.02` | Bracket take profit above the estimated entry price |
| `RISK_LOOKBACK_BARS` | `20` | Number of candles used for support/resistance exits |
| `SIMULATION_HORIZON_BARS` | `5` | Number of timeframe bars sampled per Markov/Monte Carlo path |

## What one run does

1. Downloads two days of one-minute bars for `TICKER` by default. This is a 1-minute strategy, not a 5-minute or 15-minute strategy. Set `DATA_INTERVAL=5m` or `DATA_INTERVAL=15m` to experiment with another candle size. If the newest bar is older than `MARKET_DATA_MAX_AGE_MINUTES`, the run stops without creating a prediction or order; this naturally avoids stale weekend and holiday signals.
2. Saves the raw data locally.
3. Drops rows made incomplete by lag and moving-average calculations.
4. Trains a new model using up to 500 recent samples, keeping the newest 20% aside.
5. Measures historical accuracy and the win rate of the model's historical buy predictions on that unseen portion.
6. Predicts the latest available feature row, which was excluded from training.
7. Appends the prediction to `data/prediction_history.csv`.
8. Produces a CALL/PUT directional label, a Markov/Monte Carlo five-bar distribution, and a range exit plan based on the selected timeframe. If buy probability and historical predicted-buy win rate pass their thresholds, and `ENABLE_TRADING=true`, sends a market buy for `TRADE_SYMBOL` and `TRADE_QTY` with attached stop-loss and take-profit exits. The exit *distance* is transferred to the current order quote; it never copies a futures price directly into a GLD order.

The validation metric is a small recent holdout, not a full backtest, and cannot guarantee future performance. A CALL/PUT label is a market-direction label, not an options order instruction: this MVP still trades the configured equity/ETF symbol. The program refuses duplicate buys, attaches a bracket stop loss and take profit to each accepted buy, and closes an existing position on a validated sell signal. A high probability or a Monte Carlo result is not a guarantee of profit. The bracket exits are based on the latest quote before the market order fills, so actual fill prices can differ.

## Trading safely

 The project can submit an order through your Alpaca account when you run it, so verify the following before running it:

1. Use an Alpaca paper account and paper API keys.
2. Confirm `ALPACA_BASE_URL` is exactly the paper endpoint.
3. Keep `ENABLE_TRADING=false` while reviewing signals.
4. Confirm `TRADE_SYMBOL=GLD` is the instrument you intend to trade, rather than the futures data ticker `GC=F`.
5. Review `STOP_LOSS_PCT`, `TAKE_PROFIT_PCT`, and `TRADE_QTY` before enabling paper orders.
6. Check the Alpaca dashboard after any submitted order.

Do not put real credentials in the repository, shell history, chat, or `config.example.env`. Switching the base URL to a live endpoint can cause real market orders and should only be done after independent testing and review.

## Limitations and next improvements

- Add a dry-run flag and require an explicit trading enable flag.
- Add backtesting and out-of-sample evaluation before trusting probabilities.
- Add spread/slippage-aware limit orders, position checks, exits, stop losses, and a maximum daily loss.
- Align the prediction instrument and order instrument, or explicitly model the relationship between them.
- Add tests for feature alignment, empty data, model persistence, and order construction.

Original repository: https://github.com/gitaukennedy/Gold.prediction
