import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import signal
import sys
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from binance_connection import binance_connection
from predictor import predictor, start_streaming, stop_streaming
from trade_executor import TradeExecutor
from logger_setup import get_logger, log_critical_error

# Load environment variables
load_dotenv()

# Get loggers
bot_logger = get_logger('bot')
trading_logger = get_logger('trading')
prediction_logger = get_logger('prediction')

class StrategyOptimizer:
    """Optimizes trading strategy parameters based on market conditions"""
    
    def __init__(self, lookback_periods: int = 100):
        self.lookback_periods = lookback_periods
        self.market_regime = 'neutral'  # 'trending', 'ranging', 'volatile'
        self.volatility_threshold = 0.02  # 2% daily volatility threshold
        
    def analyze_market_regime(self, symbol: str) -> str:
        """Determine current market regime using multiple indicators"""
        try:
            # Get historical data
            klines = binance_connection.get_historical_klines(
                symbol, '1d', self.lookback_periods
            )
            if not klines:
                return self.market_regime
            
            # Convert to DataFrame
            df = pd.DataFrame([{
                'close': float(k['close']),
                'high': float(k['high']),
                'low': float(k['low']),
                'volume': float(k['volume'])
            } for k in klines])
            
            # Calculate indicators
            # 1. Trend strength (ADX)
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            
            # 2. Volatility
            returns = df['close'].pct_change()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # 3. Range analysis
            recent_highs = df['high'].rolling(20).max()
            recent_lows = df['low'].rolling(20).min()
            price_range = (recent_highs - recent_lows) / recent_lows
            
            # Determine regime
            if volatility > self.volatility_threshold:
                regime = 'volatile'
            elif price_range.iloc[-1] < 0.05:  # 5% range
                regime = 'ranging'
            else:
                regime = 'trending'
            
            self.market_regime = regime
            return regime
            
        except Exception as e:
            bot_logger.error(f"Error analyzing market regime: {e}")
            return self.market_regime
    
    def optimize_parameters(self, symbol: str, current_params: Dict) -> Dict:
        """Adjust trading parameters based on market regime"""
        try:
            regime = self.analyze_market_regime(symbol)
            optimized = current_params.copy()
            
            if regime == 'volatile':
                # Reduce risk in volatile markets
                optimized['position_size_pct'] *= 0.75
                optimized['stop_loss_pct'] *= 1.5
                optimized['take_profit_pct'] *= 1.2
                optimized['min_confidence'] = 0.70  # Require higher confidence
                
            elif regime == 'ranging':
                # Tighter ranges for sideways markets
                optimized['stop_loss_pct'] *= 0.8
                optimized['take_profit_pct'] *= 0.8
                optimized['min_confidence'] = 0.65
                
            elif regime == 'trending':
                # Maximize trend following
                optimized['take_profit_pct'] *= 1.5
                optimized['stop_loss_pct'] *= 0.9
                optimized['min_confidence'] = 0.60
            
            bot_logger.info(
                f"Optimized parameters for {symbol} ({regime} market):"
                f"\n- Position size: {optimized['position_size_pct']:.1%}"
                f"\n- Stop loss: {optimized['stop_loss_pct']:.1%}"
                f"\n- Take profit: {optimized['take_profit_pct']:.1%}"
                f"\n- Min confidence: {optimized['min_confidence']:.2f}"
            )
            
            return optimized
            
        except Exception as e:
            bot_logger.error(f"Error optimizing parameters: {e}")
            return current_params

class EnhancedTradingBot:
    def __init__(self,
                 symbols: List[str],
                 initial_capital: float = 10000.0,
                 position_size_pct: float = 0.01,
                 min_confidence: float = 0.60,
                 stop_loss_pct: float = 0.02,
                 take_profit_pct: float = 0.06,
                 max_positions: int = 3,
                 max_daily_trades: int = 10):
        """Initialize the enhanced trading bot with strategy optimization"""
        try:
            bot_logger.info("Initializing enhanced trading bot...")
            
            # Store configuration
            self.symbols = symbols
            self.running = False
            self.base_params = {
                'position_size_pct': position_size_pct,
                'min_confidence': min_confidence,
                'stop_loss_pct': stop_loss_pct,
                'take_profit_pct': take_profit_pct
            }
            
            # Initialize strategy optimizer
            self.optimizer = StrategyOptimizer()
            
            # Initialize trade executor with base parameters
            self.executor = TradeExecutor(
                symbols=symbols,
                initial_capital=initial_capital,
                position_size_pct=position_size_pct,
                min_confidence=min_confidence,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                max_positions=max_positions,
                max_daily_trades=max_daily_trades
            )
            
            # Set up signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            # Initialize performance tracking
            self.performance_metrics = {
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'total_pnl': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0
            }
            
            bot_logger.info("Enhanced trading bot initialized successfully")
            
        except Exception as e:
            bot_logger.error(f"Error initializing enhanced trading bot: {e}")
            raise
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        bot_logger.info("Shutdown signal received, stopping bot...")
        self.stop()
        sys.exit(0)
    
    def update_performance_metrics(self, trade_result: Dict):
        """Update performance tracking metrics"""
        try:
            self.performance_metrics['trades'] += 1
            if trade_result['pnl'] > 0:
                self.performance_metrics['wins'] += 1
            else:
                self.performance_metrics['losses'] += 1
            
            self.performance_metrics['total_pnl'] += trade_result['pnl']
            self.performance_metrics['win_rate'] = (
                self.performance_metrics['wins'] / self.performance_metrics['trades']
                if self.performance_metrics['trades'] > 0 else 0.0
            )
            
            # Update max drawdown
            if trade_result['pnl'] < 0:
                current_drawdown = abs(trade_result['pnl'])
                self.performance_metrics['max_drawdown'] = max(
                    self.performance_metrics['max_drawdown'],
                    current_drawdown
                )
            
        except Exception as e:
            bot_logger.error(f"Error updating performance metrics: {e}")
    
    def handle_prediction(self, prediction: Dict):
        """Handle new predictions with optimized parameters"""
        try:
            if not prediction['success']:
                bot_logger.error(f"Prediction error: {prediction.get('error', 'Unknown error')}")
                return
            
            symbol = prediction['symbol']
            
            # Optimize parameters based on market conditions
            optimized_params = self.optimizer.optimize_parameters(
                symbol, self.base_params
            )
            
            # Update executor parameters for this trade
            self.executor.update_parameters(symbol, optimized_params)
            
            # Execute trade with optimized parameters
            if prediction['confidence'] >= optimized_params['min_confidence']:
                trade_result = self.executor.execute_scheduled_trades()
                if trade_result:
                    self.update_performance_metrics(trade_result)
            
        except Exception as e:
            error_info = {
                'error_type': 'Prediction Handler Error',
                'error_message': str(e),
                'component': 'EnhancedTradingBot',
                'symbol': prediction.get('symbol', 'Unknown')
            }
            log_critical_error(bot_logger, error_info)
    
    def log_performance_summary(self):
        """Log detailed performance metrics"""
        try:
            bot_logger.info(
                f"Performance Summary:"
                f"\n- Total Trades: {self.performance_metrics['trades']}"
                f"\n- Win Rate: {self.performance_metrics['win_rate']:.2%}"
                f"\n- Total PnL: ${self.performance_metrics['total_pnl']:.2f}"
                f"\n- Max Drawdown: ${self.performance_metrics['max_drawdown']:.2f}"
                f"\n- Risk-Adjusted Return: {self.calculate_risk_adjusted_return():.2f}"
            )
        except Exception as e:
            bot_logger.error(f"Error logging performance summary: {e}")
    
    def calculate_risk_adjusted_return(self) -> float:
        """Calculate Sharpe-like ratio for risk-adjusted returns"""
        try:
            if self.performance_metrics['max_drawdown'] == 0:
                return 0.0
            
            return (
                self.performance_metrics['total_pnl'] /
                self.performance_metrics['max_drawdown']
            )
        except Exception as e:
            bot_logger.error(f"Error calculating risk-adjusted return: {e}")
            return 0.0
    
    def start(self):
        """Start the enhanced trading bot"""
        try:
            if self.running:
                bot_logger.warning("Trading bot is already running")
                return
            
            bot_logger.info("Starting enhanced trading bot...")
            
            # Start trade executor
            self.executor.scheduler.start()
            bot_logger.info("Trade executor scheduler started")
            
            # Start streaming for each symbol
            for symbol in self.symbols:
                # Analyze initial market regime
                regime = self.optimizer.analyze_market_regime(symbol)
                bot_logger.info(f"Initial market regime for {symbol}: {regime}")
                
                # Start streaming with optimized parameters
                success = start_streaming(symbol, self.handle_prediction)
                if success:
                    bot_logger.info(f"Started streaming for {symbol}")
                else:
                    bot_logger.error(f"Failed to start streaming for {symbol}")
            
            self.running = True
            bot_logger.info("Enhanced trading bot started successfully")
            
            # Keep the main thread running
            while self.running:
                time.sleep(60)  # Check every minute
                self.log_performance_summary()
                
        except Exception as e:
            bot_logger.error(f"Error starting trading bot: {e}")
            self.stop()
            raise
    
    def stop(self):
        """Stop the enhanced trading bot"""
        try:
            if not self.running:
                return
            
            bot_logger.info("Stopping enhanced trading bot...")
            
            # Stop streaming for each symbol
            for symbol in self.symbols:
                success = stop_streaming(symbol)
                if success:
                    bot_logger.info(f"Stopped streaming for {symbol}")
                else:
                    bot_logger.error(f"Failed to stop streaming for {symbol}")
            
            # Stop trade executor scheduler
            if self.executor.scheduler.running:
                self.executor.scheduler.shutdown()
                bot_logger.info("Trade executor scheduler stopped")
            
            # Log final performance summary
            self.log_performance_summary()
            
            self.running = False
            bot_logger.info("Enhanced trading bot stopped successfully")
            
        except Exception as e:
            bot_logger.error(f"Error stopping trading bot: {e}")
            raise

def main():
    # Trading pairs (from backtest results)
    symbols = [
        'AMBUSDT',  # Best performer: 105.58% return, 6.60 Sharpe
        'ARUSDT',   # Second best: 98.60% return, 6.78 Sharpe
    ]
    
    try:
        # Initialize and start enhanced trading bot
        bot = EnhancedTradingBot(
            symbols=symbols,
            initial_capital=10000.0,
            position_size_pct=0.005,    # Reduced to 0.5% of capital per trade
            min_confidence=0.70,        # Increased to 70% prediction confidence
            stop_loss_pct=0.01,         # Tighter 1% stop loss
            take_profit_pct=0.03,       # Lower 3% take profit for more frequent wins
            max_positions=1,            # Reduced to 1 position at a time
            max_daily_trades=5          # Reduced to 5 trades per day
        )
        
        bot_logger.info("=" * 50)
        bot_logger.info("Starting Enhanced Binance Trading Bot")
        bot_logger.info("Mode: TESTNET (Paper Trading)")
        bot_logger.info(f"Trading pairs: {', '.join(symbols)}")
        bot_logger.info("=" * 50)
        
        # Start the bot
        bot.start()
        
    except Exception as e:
        bot_logger.error(f"Fatal error in trading bot: {e}")
        bot_logger.critical(f"Bot shutdown due to fatal error: {e}")
        raise

if __name__ == "__main__":
    main() 