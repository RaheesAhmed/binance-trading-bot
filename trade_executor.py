from binance_connection import binance_connection
from predictor import make_prediction
from logger_setup import get_logger, log_trade_execution, log_trade_exit, log_critical_error
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED
import numpy as np
import logging

# Load environment variables
load_dotenv()

# Get loggers
trading_logger = get_logger('trading')
prediction_logger = get_logger('prediction')
binance_logger = get_logger('binance')

class RiskManager:
    def __init__(self,
                 initial_capital: float,
                 max_daily_loss_pct: float = 0.02,    # 2% max daily loss
                 max_position_size_pct: float = 0.05,  # 5% max position size
                 risk_per_trade_pct: float = 0.01,     # 1% risk per trade
                 max_correlated_positions: int = 2,     # Max similar positions
                 volatility_lookback: int = 20):        # Days for volatility calc
        
        self.initial_capital = initial_capital
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_position_size_pct = max_position_size_pct
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_correlated_positions = max_correlated_positions
        self.volatility_lookback = volatility_lookback
        
        # Track daily performance
        self.daily_pnl = 0.0
        self.last_reset = datetime.now().date()
        
        trading_logger.info(
            f"Risk Manager initialized with:"
            f"\n- Max daily loss: {max_daily_loss_pct:.1%}"
            f"\n- Max position size: {max_position_size_pct:.1%}"
            f"\n- Risk per trade: {risk_per_trade_pct:.1%}"
        )
    
    def calculate_position_size(self,
                              symbol: str,
                              entry_price: float,
                              stop_loss_price: float,
                              account_balance: float) -> float:
        """Calculate position size based on risk parameters and volatility"""
        try:
            # Get historical volatility
            volatility = self._calculate_volatility(symbol)
            
            # Calculate risk amount (1% of current balance)
            risk_amount = account_balance * self.risk_per_trade_pct
            
            # Calculate position size based on stop loss distance
            stop_loss_distance = abs(entry_price - stop_loss_price) / entry_price
            base_position_size = risk_amount / stop_loss_distance
            
            # Adjust for volatility (reduce size for high volatility)
            volatility_factor = 1.0 / (1.0 + volatility)
            position_size = base_position_size * volatility_factor
            
            # Apply maximum position size limit
            max_position = account_balance * self.max_position_size_pct
            position_size = min(position_size, max_position)
            
            trading_logger.info(
                f"Position size calculation for {symbol}:"
                f"\n- Risk amount: ${risk_amount:.2f}"
                f"\n- Stop loss distance: {stop_loss_distance:.2%}"
                f"\n- Volatility factor: {volatility_factor:.2f}"
                f"\n- Final position size: ${position_size:.2f}"
            )
            
            return position_size
            
        except Exception as e:
            trading_logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def _calculate_volatility(self, symbol: str) -> float:
        """Calculate historical volatility for risk adjustment"""
        try:
            # Get historical klines
            klines = binance_connection.get_historical_klines(
                symbol, '1d', self.volatility_lookback
            )
            
            if not klines:
                return 1.0  # Default to high volatility if no data
            
            # Calculate daily returns
            prices = [float(k['close']) for k in klines]
            returns = np.diff(np.log(prices))
            
            # Calculate annualized volatility
            daily_vol = np.std(returns)
            annual_vol = daily_vol * np.sqrt(365)
            
            return annual_vol
            
        except Exception as e:
            trading_logger.error(f"Error calculating volatility: {e}")
            return 1.0
    
    def can_open_position(self,
                         symbol: str,
                         active_positions: Dict,
                         predicted_direction: int) -> bool:
        """Check if new position can be opened based on risk rules"""
        try:
            # Check daily loss limit
            if self.daily_pnl <= -(self.initial_capital * self.max_daily_loss_pct):
                trading_logger.warning("Daily loss limit reached, no new positions allowed")
                return False
            
            # Check correlation limits
            similar_positions = sum(
                1 for pos in active_positions.values()
                if pos['side'] == ('BUY' if predicted_direction == 1 else 'SELL')
            )
            
            if similar_positions >= self.max_correlated_positions:
                trading_logger.warning(
                    f"Maximum correlated positions ({self.max_correlated_positions}) reached"
                )
                return False
            
            return True
            
        except Exception as e:
            trading_logger.error(f"Error checking position limits: {e}")
            return False
    
    def update_daily_pnl(self, trade_pnl: float):
        """Update daily PnL tracking"""
        try:
            current_date = datetime.now().date()
            
            # Reset daily PnL if new day
            if current_date != self.last_reset:
                self.daily_pnl = 0.0
                self.last_reset = current_date
            
            self.daily_pnl += trade_pnl
            
            if self.daily_pnl <= -(self.initial_capital * self.max_daily_loss_pct):
                trading_logger.warning(
                    f"Daily loss limit reached: ${self.daily_pnl:.2f}"
                )
            
        except Exception as e:
            trading_logger.error(f"Error updating daily PnL: {e}")

class TradeExecutor:
    def __init__(self,
                 symbols: List[str],
                 initial_capital: float = 10000.0,
                 position_size_pct: float = 0.01,
                 min_confidence: float = 0.60,
                 stop_loss_pct: float = 0.02,
                 take_profit_pct: float = 0.06,
                 max_positions: int = 3,
                 max_daily_trades: int = 10):
        
        # Ensure we're in testnet mode
        if not binance_connection.use_testnet:
            raise ValueError("Must use testnet for paper trading")
        
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.min_confidence = min_confidence
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_positions = max_positions
        self.max_daily_trades = max_daily_trades
        
        # Initialize risk manager
        self.risk_manager = RiskManager(
            initial_capital=initial_capital,
            max_daily_loss_pct=0.02,     # 2% max daily loss
            max_position_size_pct=0.05,   # 5% max position size
            risk_per_trade_pct=0.01,      # 1% risk per trade
            max_correlated_positions=2     # Max 2 similar positions
        )
        
        # Initialize tracking
        self.active_positions = {}  # symbol -> position info
        self.daily_trades = 0       # Track daily trade count
        self.last_trade_reset = datetime.now().date()
        
        # Initialize scheduler
        self.scheduler = BackgroundScheduler()
        self.setup_scheduler()
        
        # Get initial balance
        self.update_balance()
        trading_logger.info(f"Initial USDT balance: {self.usdt_balance}")
        
        trading_logger.warning("=" * 50)
        trading_logger.warning("TESTNET MODE - Paper Trading Only")
        trading_logger.warning("=" * 50)
    
    def setup_scheduler(self):
        """Set up the APScheduler with jobs and error handling"""
        # Add job listener for logging
        def job_listener(event):
            if event.exception:
                trading_logger.error(f"Job failed: {event.exception}")
                trading_logger.critical(f"Critical error in scheduled job: {event.exception}")
            else:
                trading_logger.info(f"Job completed successfully: {event.job_id}")
        
        self.scheduler.add_listener(job_listener, EVENT_JOB_ERROR | EVENT_JOB_EXECUTED)
        
        # Schedule trading job to run every hour
        self.scheduler.add_job(
            func=self.execute_scheduled_trades,
            trigger=CronTrigger(minute=0),  # Run at the start of every hour
            id='hourly_trading',
            name='Hourly Trading Execution',
            misfire_grace_time=300  # Allow 5 minutes grace time for missed executions
        )
        
        # Schedule position management to run every 5 minutes
        self.scheduler.add_job(
            func=self.manage_positions,
            trigger=CronTrigger(minute='*/5'),  # Run every 5 minutes
            id='position_management',
            name='Position Management',
            misfire_grace_time=60
        )
        
        # Schedule daily reset job at midnight
        self.scheduler.add_job(
            func=self.daily_reset,
            trigger=CronTrigger(hour=0, minute=0),  # Run at midnight
            id='daily_reset',
            name='Daily Counter Reset',
            misfire_grace_time=3600  # Allow 1 hour grace time for missed reset
        )
        
        trading_logger.info("Scheduler jobs configured successfully")
    
    def daily_reset(self):
        """Reset daily counters and risk metrics"""
        try:
            # Reset trade executor counters
            self.daily_trades = 0
            self.last_trade_reset = datetime.now().date()
            
            # Reset risk manager counters
            self.risk_manager.daily_pnl = 0.0
            self.risk_manager.last_reset = datetime.now().date()
            
            # Log daily summary before reset
            self.log_daily_summary()
            
            trading_logger.info(
                "Daily reset completed:"
                "\n- Trade counter reset"
                "\n- PnL counter reset"
                "\n- Risk metrics reset"
            )
            
        except Exception as e:
            trading_logger.error(f"Error in daily reset: {e}")
            trading_logger.critical("Failed to reset daily counters")
    
    def log_daily_summary(self):
        """Log summary of the day's trading activity"""
        try:
            # Calculate daily statistics
            total_positions = len(self.active_positions)
            daily_pnl = self.risk_manager.daily_pnl
            pnl_percentage = (daily_pnl / self.initial_capital) * 100
            
            trading_logger.info(
                f"Daily Trading Summary:"
                f"\n- Total Trades: {self.daily_trades}"
                f"\n- Active Positions: {total_positions}"
                f"\n- Daily PnL: ${daily_pnl:.2f} ({pnl_percentage:.2f}%)"
                f"\n- Account Balance: ${self.usdt_balance:.2f}"
            )
            
            # Log individual position details
            if self.active_positions:
                trading_logger.info("Active Positions:")
                for symbol, position in self.active_positions.items():
                    unrealized_pnl = position.get('unrealized_pnl', 0.0)
                    holding_time = datetime.now() - position['entry_time']
                    
                    trading_logger.info(
                        f"- {symbol}:"
                        f"\n  Side: {position['side']}"
                        f"\n  Entry Price: {position['entry_price']:.8f}"
                        f"\n  Current Price: {position.get('current_price', 'N/A')}"
                        f"\n  Unrealized PnL: ${unrealized_pnl:.2f}"
                        f"\n  Holding Time: {holding_time}"
                    )
            
        except Exception as e:
            trading_logger.error(f"Error logging daily summary: {e}")
    
    def execute_scheduled_trades(self):
        """Execute trades on schedule"""
        try:
            trading_logger.info("Starting scheduled trading execution")
            
            # Look for new trading opportunities
            for symbol in self.symbols:
                if symbol in self.active_positions:
                    continue
                
                if len(self.active_positions) >= self.max_positions:
                    trading_logger.info("Maximum positions reached, skipping trading")
                    break
                
                prediction = make_prediction(symbol)
                if prediction['success']:
                    self.execute_trade(symbol, prediction)
                
            trading_logger.info("Completed scheduled trading execution")
            
        except Exception as e:
            trading_logger.error(f"Error in scheduled trading execution: {e}")
            trading_logger.critical(f"Critical error during scheduled trading: {e}")
    
    def update_balance(self) -> float:
        """Update and return current USDT balance"""
        try:
            account = binance_connection.client.get_account()
            self.usdt_balance = next(
                (float(asset['free']) for asset in account['balances'] if asset['asset'] == 'USDT'),
                0.0
            )
            return self.usdt_balance
        except Exception as e:
            logging.error(f"Error updating balance: {e}")
            return 0.0
    
    def execute_trade(self, symbol: str, prediction: Dict) -> bool:
        """Execute trade based on prediction and risk management"""
        try:
            # Skip if prediction confidence is too low
            if prediction['confidence'] < self.min_confidence:
                return False
            
            # Check risk management rules
            if not self.risk_manager.can_open_position(
                symbol,
                self.active_positions,
                prediction['prediction']
            ):
                return False
            
            # Get current price
            current_price = prediction['current_price']
            
            # Calculate stop loss and take profit levels
            stop_loss = current_price * (1 - self.stop_loss_pct) if prediction['prediction'] == 1 else \
                       current_price * (1 + self.stop_loss_pct)
            
            take_profit = current_price * (1 + self.take_profit_pct) if prediction['prediction'] == 1 else \
                         current_price * (1 - self.take_profit_pct)
            
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                symbol,
                current_price,
                stop_loss,
                self.usdt_balance
            )
            
            if position_size <= 0:
                trading_logger.warning(f"Invalid position size calculated for {symbol}")
                return False
            
            # Execute the trade
            side = 'BUY' if prediction['prediction'] == 1 else 'SELL'
            
            # Store position information
            self.active_positions[symbol] = {
                'side': side,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'position_size': position_size,
                'entry_time': datetime.now()
            }
            
            # Log trade execution with email notification
            trade_info = {
                'symbol': symbol,
                'side': side,
                'entry_price': current_price,
                'position_size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': prediction['confidence']
            }
            log_trade_execution(trading_logger, trade_info)
            
            return True
            
        except Exception as e:
            error_info = {
                'error_type': 'Trade Execution Error',
                'error_message': str(e),
                'component': 'TradeExecutor',
                'symbol': symbol,
                'additional_info': f"Prediction: {prediction}"
            }
            log_critical_error(trading_logger, error_info)
            return False
    
    def manage_positions(self):
        """Manage existing positions with risk management"""
        try:
            for symbol in list(self.active_positions.keys()):
                position = self.active_positions[symbol]
                
                # Get current price
                current_price = binance_connection.get_symbol_price(symbol)
                if not current_price:
                    continue
                
                # Calculate unrealized PnL
                if position['side'] == 'BUY':
                    pnl = (current_price - position['entry_price']) * position['position_size']
                else:
                    pnl = (position['entry_price'] - current_price) * position['position_size']
                
                # Check stop loss and take profit
                if (position['side'] == 'BUY' and current_price <= position['stop_loss']) or \
                   (position['side'] == 'SELL' and current_price >= position['stop_loss']):
                    position['exit_reason'] = 'Stop Loss'
                    self.close_position(symbol)
                    self.risk_manager.update_daily_pnl(pnl)
                    
                elif (position['side'] == 'BUY' and current_price >= position['take_profit']) or \
                     (position['side'] == 'SELL' and current_price <= position['take_profit']):
                    position['exit_reason'] = 'Take Profit'
                    self.close_position(symbol)
                    self.risk_manager.update_daily_pnl(pnl)
                
                # Check time-based exit (24 hours)
                elif datetime.now() - position['entry_time'] > timedelta(hours=24):
                    position['exit_reason'] = 'Time Exit'
                    self.close_position(symbol)
                    self.risk_manager.update_daily_pnl(pnl)
                
                # Update position information
                position['current_price'] = current_price
                position['unrealized_pnl'] = pnl
                
        except Exception as e:
            error_info = {
                'error_type': 'Position Management Error',
                'error_message': str(e),
                'component': 'TradeExecutor'
            }
            log_critical_error(trading_logger, error_info)
    
    def close_position(self, symbol: str):
        """Close a position and update tracking"""
        try:
            position = self.active_positions[symbol]
            current_price = binance_connection.get_symbol_price(symbol)
            
            if position['side'] == 'BUY':
                pnl = (current_price - position['entry_price']) * position['position_size']
            else:
                pnl = (position['entry_price'] - current_price) * position['position_size']
            
            # Log trade exit with email notification
            exit_info = {
                'symbol': symbol,
                'side': position['side'],
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'pnl': pnl,
                'holding_time': str(datetime.now() - position['entry_time']),
                'exit_reason': position.get('exit_reason', 'Manual Exit')
            }
            log_trade_exit(trading_logger, exit_info)
            
            del self.active_positions[symbol]
            self.risk_manager.update_daily_pnl(pnl)
            
        except Exception as e:
            error_info = {
                'error_type': 'Position Close Error',
                'error_message': str(e),
                'component': 'TradeExecutor',
                'symbol': symbol
            }
            log_critical_error(trading_logger, error_info)
    
    def run(self):
        """Start the trading bot with scheduler"""
        try:
            trading_logger.info("Starting trading bot with scheduler...")
            
            # Start the scheduler
            self.scheduler.start()
            trading_logger.info("Scheduler started successfully")
            
            # Keep the script running
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                trading_logger.info("Shutting down trading bot...")
                self.scheduler.shutdown()
                
        except Exception as e:
            trading_logger.error(f"Error running trading bot: {e}")
            trading_logger.critical(f"Critical error in trading bot: {e}")
            self.scheduler.shutdown()
            raise

def main():
    # Trading pairs (from backtest results)
    symbols = [
        'AMBUSDT',  # Best performer: 105.58% return, 6.60 Sharpe
        'ARUSDT',   # Second best: 98.60% return, 6.78 Sharpe
        'AIUSDT',   # Third best: 97.06% return, 5.98 Sharpe
    ]
    
    try:
        # Initialize and run trade executor
        executor = TradeExecutor(
            symbols=symbols,
            initial_capital=10000.0,
            position_size_pct=0.01,     # 1% of capital per trade
            min_confidence=0.60,        # Minimum 60% prediction confidence
            stop_loss_pct=0.02,         # 2% stop loss
            take_profit_pct=0.06,       # 6% take profit
            max_positions=3,            # Maximum 3 simultaneous positions
            max_daily_trades=10         # Maximum 10 trades per day
        )
        
        trading_logger.info("Starting trade executor with scheduler...")
        executor.run()
        
    except Exception as e:
        trading_logger.error(f"Fatal error in trade executor: {e}")
        trading_logger.critical(f"Bot shutdown due to fatal error: {e}")
        raise

if __name__ == "__main__":
    main() 