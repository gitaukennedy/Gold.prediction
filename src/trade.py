import os

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestTradeRequest
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderClass, OrderSide, OrderType, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, TakeProfitRequest
except Exception:
    TradingClient = None
    StockHistoricalDataClient = None


def get_client():
    key = os.getenv('APCA_API_KEY_ID') or os.getenv('ALPACA_API_KEY')
    secret = os.getenv('APCA_API_SECRET_KEY') or os.getenv('ALPACA_SECRET_KEY')
    base = os.getenv('APCA_API_BASE_URL') or os.getenv('ALPACA_BASE_URL') or 'https://paper-api.alpaca.markets'
    if not key or not secret:
        raise EnvironmentError('Alpaca API keys not set in environment')
    if TradingClient is None:
        raise RuntimeError('alpaca-py not installed or failed to import')
    paper = 'paper-api.alpaca.markets' in base
    return TradingClient(api_key=key, secret_key=secret, paper=paper, url_override=base)


def get_position(symbol: str):
    client = get_client()
    return next((position for position in client.get_all_positions()
                 if position.symbol == symbol), None)


def get_latest_price(symbol: str) -> float:
    key = os.getenv('APCA_API_KEY_ID') or os.getenv('ALPACA_API_KEY')
    secret = os.getenv('APCA_API_SECRET_KEY') or os.getenv('ALPACA_SECRET_KEY')
    if StockHistoricalDataClient is None:
        raise RuntimeError('alpaca-py market data client is unavailable')
    data_client = StockHistoricalDataClient(key, secret)
    latest_trade = data_client.get_stock_latest_trade(
        StockLatestTradeRequest(symbol_or_symbols=symbol)
    )
    price = latest_trade[symbol].price
    if price <= 0:
        raise RuntimeError(f'Invalid latest price for {symbol}: {price}')
    return float(price)


def place_bracket_buy(symbol: str, qty: int = 1, stop_distance_pct: float = None,
                      target_distance_pct: float = None):
    if get_position(symbol) is not None:
        raise RuntimeError(f'Position already exists for {symbol}; buy skipped')
    stop_loss_pct = stop_distance_pct or float(os.getenv('STOP_LOSS_PCT', '0.01'))
    take_profit_pct = target_distance_pct or float(os.getenv('TAKE_PROFIT_PCT', '0.02'))
    if not 0 < stop_loss_pct < 1 or not 0 < take_profit_pct < 1:
        raise ValueError('Risk distances must be between 0 and 1')
    entry_price = get_latest_price(symbol)
    stop_price = round(entry_price * (1 - stop_loss_pct), 2)
    take_profit_price = round(entry_price * (1 + take_profit_pct), 2)
    order = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY,
        type=OrderType.MARKET,
        time_in_force=TimeInForce.GTC,
        order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(limit_price=take_profit_price),
        stop_loss=StopLossRequest(stop_price=stop_price),
    )
    return get_client().submit_order(order_data=order)


def close_position(symbol: str):
    position = get_position(symbol)
    if position is None:
        return None
    return get_client().close_position(symbol)


def place_order(symbol: str, qty: int = 1, side: str = 'buy', order_type: str = 'market', time_in_force: str = 'gtc'):
    """Backward-compatible simple order helper for non-bracket use."""
    if order_type != 'market':
        raise ValueError('Only market orders are supported')
    order = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide(side.lower()),
        type=OrderType.MARKET,
        time_in_force=TimeInForce(time_in_force.lower()),
    )
    return get_client().submit_order(order_data=order)
