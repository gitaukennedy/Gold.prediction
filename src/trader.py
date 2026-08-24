"""Automated trading logic and execution"""
import pandas as pd
import logging
from typing import Dict, List, Any
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradeLog:
    """Maintain trading history"""
    
    def __init__(self):
        self.trades = []
    
    def add_trade(self, trade: Dict):
        """Add trade to log"""
        self.trades.append(trade)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert trades to DataFrame"""
        return pd.DataFrame(self.trades)
    
    def get_summary(self) -> Dict:
        """Get trading summary statistics"""
        if not self.trades:
            return {'total_trades': 0}
        
        df = self.to_dataframe()
        winning_trades = df[df['Profit'] > 0]
        losing_trades = df[df['Profit'] < 0]
        
        return {
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.trades) if len(self.trades) > 0 else 0,
            'total_profit': df['Profit'].sum(),
            'avg_profit': df['Profit'].mean(),
            'max_profit': df['Profit'].max(),
            'max_loss': df['Profit'].min()
        }


class AutomatedTrader:
    """Execute trades based on signals"""
    
    def __init__(self, initial_capital: float = 10000, position_size: float = 1000):
        """
        Initialize trader
        
        Args:
            initial_capital: Starting capital in USD
            position_size: Capital per trade in USD
        """
        self.capital = initial_capital
        self.position_size = position_size
        self.positions = []  # Open positions
        self.trade_log = TradeLog()
        self.cash = initial_capital
        
    def execute_trade(self, signal: Dict, current_price: float, timestamp: datetime) -> Dict:
        """
        Execute trade based on signal
        
        Args:
            signal: Trading signal from predictor
            current_price: Current gold price
            timestamp: Trade timestamp
            
        Returns:
            Trade execution details
        """
        trade_result = {
            'timestamp': timestamp,
            'signal': signal['signal'],
            'entry_price': current_price,
            'quantity': 0,
            'status': 'REJECTED',
            'reason': ''
        }
        
        try:
            if signal['signal'] == 'BUY':
                if self.cash >= self.position_size:
                    quantity = self.position_size / current_price
                    
                    self.positions.append({
                        'type': 'LONG',
                        'entry_price': current_price,
                        'quantity': quantity,
                        'entry_time': timestamp,
                        'stop_loss': current_price * 0.98,  # 2% stop loss
                        'take_profit': current_price * 1.03  # 3% take profit
                    })
                    
                    self.cash -= self.position_size
                    trade_result['quantity'] = quantity
                    trade_result['status'] = 'EXECUTED'
                    logger.info(f"BUY: {quantity:.4f} units at ${current_price:.2f}")
                else:
                    trade_result['reason'] = 'Insufficient capital'
                    
            elif signal['signal'] == 'SELL':
                if self.positions:
                    # Close first open position
                    position = self.positions.pop(0)
                    profit = (current_price - position['entry_price']) * position['quantity']
                    
                    self.cash += (current_price * position['quantity'])
                    
                    trade = {
                        'Entry_Time': position['entry_time'],
                        'Exit_Time': timestamp,
                        'Entry_Price': position['entry_price'],
                        'Exit_Price': current_price,
                        'Quantity': position['quantity'],
                        'Profit': profit,
                        'Return%': (profit / self.position_size) * 100
                    }
                    self.trade_log.add_trade(trade)
                    
                    trade_result['quantity'] = position['quantity']
                    trade_result['status'] = 'EXECUTED'
                    trade_result['profit'] = profit
                    logger.info(f"SELL: {position['quantity']:.4f} units at ${current_price:.2f} | Profit: ${profit:.2f}")
                else:
                    trade_result['reason'] = 'No open positions'
            
            else:
                trade_result['reason'] = 'Hold signal - no action'
                
        except Exception as e:
            trade_result['status'] = 'ERROR'
            trade_result['reason'] = str(e)
            logger.error(f"Trade execution error: {e}")
        
        return trade_result
    
    def check_stop_loss_take_profit(self, current_price: float, timestamp: datetime) -> List[Dict]:
        """
        Check and execute stop loss / take profit orders
        
        Args:
            current_price: Current gold price
            timestamp: Current timestamp
            
        Returns:
            List of executed SL/TP trades
        """
        executions = []
        
        for i, position in enumerate(self.positions[:]):
            exit_reason = None
            
            if current_price <= position['stop_loss']:
                exit_reason = 'STOP_LOSS'
            elif current_price >= position['take_profit']:
                exit_reason = 'TAKE_PROFIT'
            
            if exit_reason:
                self.positions.remove(position)
                profit = (current_price - position['entry_price']) * position['quantity']
                self.cash += (current_price * position['quantity'])
                
                trade = {
                    'Entry_Time': position['entry_time'],
                    'Exit_Time': timestamp,
                    'Entry_Price': position['entry_price'],
                    'Exit_Price': current_price,
                    'Quantity': position['quantity'],
                    'Profit': profit,
                    'Return%': (profit / self.position_size) * 100,
                    'Exit_Reason': exit_reason
                }
                self.trade_log.add_trade(trade)
                executions.append(trade)
                logger.info(f"{exit_reason} Triggered at ${current_price:.2f}")
        
        return executions
    
    def get_portfolio_value(self, current_price: float) -> float:
        """Calculate current portfolio value"""
        position_value = sum(pos['quantity'] * current_price for pos in self.positions)
        return self.cash + position_value
    
    def get_summary(self) -> Dict:
        """Get trading summary"""
        return self.trade_log.get_summary()


def main():
    """Test trader"""
    trader = AutomatedTrader(initial_capital=10000, position_size=1000)
    
    # Simulate trades
    signal_buy = {'signal': 'BUY', 'confidence': 75}
    trade1 = trader.execute_trade(signal_buy, 2000, pd.Timestamp.now())
    print("Trade 1:", trade1)
    
    signal_sell = {'signal': 'SELL', 'confidence': 80}
    trade2 = trader.execute_trade(signal_sell, 2050, pd.Timestamp.now())
    print("Trade 2:", trade2)
    
    print("\nTrading Summary:", trader.get_summary())


if __name__ == "__main__":
    main()
