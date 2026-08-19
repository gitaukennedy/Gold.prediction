import os

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest
except Exception:
    TradingClient = None


def get_client():
    key = os.getenv('APCA_API_KEY_ID') or os.getenv('ALPACA_API_KEY')
    secret = os.getenv('APCA_API_SECRET_KEY') or os.getenv('ALPACA_SECRET_KEY')
    base = os.getenv('APCA_API_BASE_URL') or os.getenv('ALPACA_BASE_URL') or 'https://paper-api.alpaca.markets'
    if not key or not secret:
        raise EnvironmentError('Alpaca API keys not set in environment')
    if TradingClient is None:
        raise RuntimeError('alpaca-py not installed or failed to import')
    paper = 'paper-api.alpaca.markets' in base
    return TradingClient(api_key=key, secret_key=secret, paper=paper)


def place_order(symbol: str, qty: int = 1, side: str = 'buy', order_type: str = 'market', time_in_force: str = 'gtc'):
    client = get_client()
    if order_type != 'market':
        raise ValueError('Only market orders are supported')
    order = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide(side.lower()),
        time_in_force=TimeInForce(time_in_force.lower()),
    )
    return client.submit_order(order_data=order)
