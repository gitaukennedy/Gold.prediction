Gold prediction — 30-minute MVP

Overview

This repository contains a 30-minute MVP for live gold prediction and trading. It fetches minute-level gold data from yfinance, trains a lightweight RandomForest to predict short-term direction, and (optionally) places trades via Alpaca (paper/live) when the model signals a buy.

Structure

- main.py                - orchestrator to fetch data, train/predict, and optionally trade
- src/fetch_data.py      - fetches minute history from yfinance
- src/features.py        - feature engineering (lags, moving averages)
- src/model.py           - trains and saves a simple model
- src/trade.py           - places orders via Alpaca (reads credentials from environment)
- models/                - saved models (created at runtime)
- requirements.txt       - Python dependencies
- config.example.env     - example environment variables

Quickstart

1. Create a Python venv and activate it.
2. pip install -r requirements.txt
3. Copy config.example.env to .env or export the variables below.

Required environment variables

- ALPACA_API_KEY or APCA_API_KEY_ID
- ALPACA_SECRET_KEY or APCA_API_SECRET_KEY
- ALPACA_BASE_URL or APCA_API_BASE_URL (optional, defaults to https://paper-api.alpaca.markets)
- (optional) TICKER (default GC=F), BUY_THRESHOLD, TRADE_QTY

Run (paper):

- python main.py

Notes and disclaimers

- This is a fast MVP: the model and features are intentionally simple for speed. Treat it as proof-of-concept only.
- Trading with live capital carries risk. Test thoroughly in paper mode before any live trading.

GitHub

Repository provided: https://github.com/gitaukennedy/Gold.prediction

# Gold.prediction
# Gold.prediction
