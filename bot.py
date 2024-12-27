import os
import time
from datetime import datetime
from typing import Dict, List
import signal
import sys
from dotenv import load_dotenv

from binance_connection import binance_connection
from predictor import predictor, start_streaming, stop_streaming
from trade_executor import TradeExecutor
from logger_setup import get_logger

# Load environment variables
load_dotenv()

# Get loggers
bot_logger = get_logger('bot')
trading_logger = get_logger('trading')
prediction_logger = get_logger('prediction')

class TradingBot:
    def __init__(self,
                 symbols: List[str],
                 initial_capital: float = 10000.0,
                 position_size_pct: float = 0.01,
                 min_confidence: float = 0.60,
                 stop_loss_pct: float = 0.02,
                 take_profit_pct: float = 0.06,
                 max_positions: int = 3,
                 max_daily_trades: int = 10):
        """Initialize the trading bot with all components"""
        try:
            bot_logger.info("Initializing trading bot...")
            
            # Store configuration
            self.symbols = symbols
            self.running = False
            
            # Initialize trade executor
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
            
            bot_logger.info("Trading bot initialized successfully")
            
        except Exception as e:
            bot_logger.error(f"Error initializing trading bot: {e}")
            raise
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        bot_logger.info("Shutdown signal received, stopping bot...")
        self.stop()
        sys.exit(0)
    
    def handle_prediction(self, prediction: Dict):
        """Handle new predictions from the streaming system"""
        try:
            if not prediction['success']:
                bot_logger.error(f"Prediction error: {prediction.get('error', 'Unknown error')}")
                return
            
            symbol = prediction['symbol']
            pred_direction = prediction['prediction']
            confidence = prediction['confidence']
            current_price = prediction['current_price']
            
            bot_logger.info(f"New prediction for {symbol}: "
                          f"{'UP' if pred_direction == 1 else 'DOWN'} "
                          f"with {confidence:.2%} confidence at {current_price}")
            
            # Check if we should execute a trade
            if confidence >= self.executor.min_confidence:
                # Execute trade through scheduler to maintain timing
                self.executor.execute_scheduled_trades()
            
        except Exception as e:
            bot_logger.error(f"Error handling prediction: {e}")
    
    def start(self):
        """Start the trading bot"""
        try:
            if self.running:
                bot_logger.warning("Trading bot is already running")
                return
            
            bot_logger.info("Starting trading bot...")
            
            # Start trade executor
            self.executor.scheduler.start()
            bot_logger.info("Trade executor scheduler started")
            
            # Start streaming for each symbol
            for symbol in self.symbols:
                success = start_streaming(symbol, self.handle_prediction)
                if success:
                    bot_logger.info(f"Started streaming for {symbol}")
                else:
                    bot_logger.error(f"Failed to start streaming for {symbol}")
            
            self.running = True
            bot_logger.info("Trading bot started successfully")
            
            # Keep the main thread running
            while self.running:
                time.sleep(1)
                
        except Exception as e:
            bot_logger.error(f"Error starting trading bot: {e}")
            self.stop()
            raise
    
    def stop(self):
        """Stop the trading bot"""
        try:
            if not self.running:
                return
            
            bot_logger.info("Stopping trading bot...")
            
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
            
            self.running = False
            bot_logger.info("Trading bot stopped successfully")
            
        except Exception as e:
            bot_logger.error(f"Error stopping trading bot: {e}")
            raise

def main():
    # Trading pairs (from backtest results)
    symbols = [
        'AMBUSDT',  # Best performer: 105.58% return, 6.60 Sharpe
        'ARUSDT',   # Second best: 98.60% return, 6.78 Sharpe
        'AIUSDT',   # Third best: 97.06% return, 5.98 Sharpe
    ]
    
    try:
        # Initialize and start trading bot
        bot = TradingBot(
            symbols=symbols,
            initial_capital=10000.0,
            position_size_pct=0.01,     # 1% of capital per trade
            min_confidence=0.60,        # Minimum 60% prediction confidence
            stop_loss_pct=0.02,         # 2% stop loss
            take_profit_pct=0.06,       # 6% take profit
            max_positions=3,            # Maximum 3 simultaneous positions
            max_daily_trades=10         # Maximum 10 trades per day
        )
        
        bot_logger.info("=" * 50)
        bot_logger.info("Starting Binance Trading Bot")
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