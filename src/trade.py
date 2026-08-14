import os

try:
    from alpaca_trade_api.rest import REST
except Exception:
    REST = None


def get_client():
    key = os.getenv('APCA_API_KEY_ID') or os.getenv('ALPACA_API_KEY')
    secret = os.getenv('APCA_API_SECRET_KEY') or os.getenv('ALPACA_SECRET_KEY')
    base = os.getenv('APCA_API_BASE_URL') or os.getenv('ALPACA_BASE_URL') or 'https://paper-api.alpaca.markets'
    if not key or not secret:
        raise EnvironmentError('Alpaca API keys not set in environment')
    if REST is None:
        raise RuntimeError('alpaca_trade_api not installed or failed to import')
    return REST(key, secret, base)


def place_order(symbol: str, qty: int = 1, side: str = 'buy', order_type: str = 'market', time_in_force: str = 'gtc'):
    client = get_client()
    return client.submit_order(symbol=symbol, qty=qty, side=side, type=order_type, time_in_force=time_in_force)
