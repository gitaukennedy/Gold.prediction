# Gold prediction and Alpaca trading MVP

This project fetches recent one-minute gold-market data, creates short-term price features, trains a small Random Forest classifier, reports the probability that the next bar will rise or fall, and can submit an Alpaca order when the buy probability passes a configured threshold.

It is a proof of concept, not a validated trading strategy. The model does not account for spread, slippage, commissions, market hours, position size, portfolio exposure, stop losses, or drawdown. Use paper trading first and review every signal yourself.

## Project flow

```text
yfinance (GC=F) -> data/latest.csv
			  |
			  v
		  feature creation
			  |
			  v
	  RandomForest train/save/load
			  |
			  v
	    buy/sell probability report
			  |
	    buy signal only -> Alpaca
```

The default market-data ticker is `GC=F` (COMEX gold futures). The default order symbol is `GLD` (the SPDR Gold Shares ETF); these are different instruments and are not interchangeable. Change them only after confirming that the symbol is supported by the configured Alpaca account.

## Repository files

### Application code

- `main.py`: Application orchestrator. Loads `.env`, fetches two days of one-minute data, prints the latest 20 rows, creates features and labels, trains on the latest 500 usable rows, loads `models/rf.pkl`, prints class probabilities, and submits a market buy when the buy probability is greater than `BUY_THRESHOLD`. Sell signals are reported but do not submit sell orders.
- `src/__init__.py`: Package marker for the `src` Python package.
- `src/fetch_data.py`: Calls `yfinance.Ticker.history()`, keeps `Open`, `High`, `Low`, `Close`, and `Volume`, removes missing rows, and writes `data/latest.csv` and `data/latest_20.csv`.
- `src/features.py`: Computes percentage returns, five lagged returns, five-bar and 15-bar moving averages, and their ratio. The label is `1` when the next close is higher than the current close, otherwise `0`.
- `src/model.py`: Creates a 50-tree `RandomForestClassifier`, fits it, saves it with `joblib` to `models/rf.pkl`, and provides the matching loader.
- `src/trade.py`: Builds an Alpaca `TradingClient` from environment credentials and submits market orders. The base URL determines paper versus live mode; the default is Alpaca paper trading.

### Configuration and operations

- `config.example.env`: Safe configuration template. Copy it to `.env`; never commit real keys.
- `requirements.txt`: Pinned or bounded Python dependencies: yfinance, pandas, scikit-learn, joblib, alpaca-py, and python-dotenv.
- `scripts/run_mvp.bat`: Windows helper that installs dependencies and runs `main.py`. Git Bash users can run the equivalent commands shown below.
- `.gitignore`: Excludes secrets, virtual environments, Python cache files, and generated model files from Git.
- `commitmsg.txt`: Text used as a commit-message note; it is not needed to run the application.

### Generated or supporting files

- `data/latest.csv`: Latest downloaded history. Replaced on each successful run and ignored only if local Git rules are extended to ignore it.
- `data/latest_20.csv`: Last 20 downloaded rows for quick inspection.
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

The script needs network access to Yahoo Finance. It needs valid Alpaca paper credentials only when a buy signal passes the threshold, because that is when `place_order()` is called.

## Configuration reference

| Variable | Default | Meaning |
| --- | --- | --- |
| `ALPACA_API_KEY` / `APCA_API_KEY_ID` | none | Alpaca API key |
| `ALPACA_SECRET_KEY` / `APCA_API_SECRET_KEY` | none | Alpaca secret |
| `ALPACA_BASE_URL` / `APCA_API_BASE_URL` | paper URL | Paper or live Alpaca endpoint |
| `TICKER` | `GC=F` | yfinance data ticker |
| `TRADE_SYMBOL` | `GLD` | Alpaca order symbol |
| `BUY_THRESHOLD` | `0.55` | Buy probability must be greater than this value |
| `SELL_THRESHOLD` | `0.55` | Sell probability threshold; currently report-only |
| `MINIMUM_WIN_RATE` | `0.52` | Minimum recent predicted-buy win rate required before a buy |
| `TRADE_QTY` | `1` | Whole-unit quantity for the buy order |
| `TRADE_SYMBOL` | `GLD` | Alpaca order symbol |
| `ENABLE_TRADING` | `false` | Must be `true` before any order can be submitted |
| `STOP_LOSS_PCT` | `0.01` | Bracket stop loss below the estimated entry price |
| `TAKE_PROFIT_PCT` | `0.02` | Bracket take profit above the estimated entry price |

## What one run does

1. Downloads two days of one-minute bars for `TICKER`.
2. Saves the raw data locally.
3. Drops rows made incomplete by lag and moving-average calculations.
4. Trains a new model using up to 500 recent samples, keeping the newest 20% aside.
5. Measures historical accuracy and the win rate of the model's historical buy predictions on that unseen portion.
6. Predicts the latest available feature row, which was excluded from training.
7. Prints buy and sell probabilities plus the historical win-rate metrics.
8. If buy probability and historical predicted-buy win rate pass their thresholds, and `ENABLE_TRADING=true`, sends a market buy for `TRADE_SYMBOL` and `TRADE_QTY` with attached stop-loss and take-profit exits. Otherwise it does not submit an order.

The validation metric is a small recent holdout, not a full backtest, and cannot guarantee future performance. The program now refuses duplicate buys, attaches a bracket stop loss and take profit to each accepted buy, and closes an existing position on a validated sell signal. A high probability is not a guarantee of profit. The bracket exits are based on the latest quote before the market order fills, so actual fill prices can differ.

## Trading safely

I cannot place a financial trade for you. The project can submit an order through your Alpaca account when you run it, so verify the following before running it:

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
