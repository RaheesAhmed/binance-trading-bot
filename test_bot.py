import os
import logging
from datetime import datetime
import time
from typing import Dict, Optional
import json

# Import all components
from binance_connection import binance_connection
from data_preprocessing import process_all_data
from model_training import train_model
from trading_execution import TradingExecutor

# Configure logging
def setup_logging():
    """Set up logging configuration"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"trading_bot_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

class TradingBot:
    def __init__(self,
                 symbols: list,
                 update_interval: int = 3600,  # 1 hour in seconds
                 retrain_interval: int = 86400,  # 24 hours in seconds
                 initial_capital: float = 10000.0,
                 position_size: float = 0.1,
                 stop_loss_pct: float = 0.02,
                 take_profit_pct: float = 0.06,
                 max_positions: int = 5):
        
        self.symbols = symbols
        self.update_interval = update_interval
        self.retrain_interval = retrain_interval
        self.trading_params = {
            'initial_capital': initial_capital,
            'position_size': position_size,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            'max_positions': max_positions
        }
        
        # Initialize last update timestamps
        self.last_data_update = 0
        self.last_model_retrain = 0
        
        # Verify Testnet connection
        if not binance_connection.use_testnet:
            raise ValueError("Must use Testnet for testing! Check .env configuration.")
        
        logging.info("Initializing Trading Bot in Testnet mode")
        self.log_account_info()
    
    def log_account_info(self):
        """Log current account information"""
        try:
            account_info = binance_connection.get_account_info()
            logging.info("Account Information:")
            logging.info(f"Total Balance: {account_info['totalBalance']:.2f} USDT")
            logging.info(f"Available Balance: {account_info['availableBalance']:.2f} USDT")
            logging.info(f"Open Orders: {account_info['openOrders']}")
        except Exception as e:
            logging.error(f"Error getting account info: {e}")
    
    def update_data(self) -> bool:
        """Update historical data and preprocess it"""
        try:
            current_time = time.time()
            
            # Check if update is needed
            if current_time - self.last_data_update < self.update_interval:
                return True
            
            logging.info("Updating historical data...")
            
            # Fetch new data for each symbol
            for symbol in self.symbols:
                logging.info(f"Fetching data for {symbol}...")
                klines = binance_connection.get_historical_klines(symbol, '1h', 1000)
                if not klines:
                    logging.error(f"Failed to fetch data for {symbol}")
                    continue
            
            # Preprocess the data
            logging.info("Processing historical data...")
            process_all_data()
            
            self.last_data_update = current_time
            logging.info("Data update completed successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error updating data: {e}")
            return False
    
    def retrain_model(self) -> bool:
        """Retrain the trading model"""
        try:
            current_time = time.time()
            
            # Check if retraining is needed
            if current_time - self.last_model_retrain < self.retrain_interval:
                return True
            
            logging.info("Retraining model...")
            train_model()
            
            self.last_model_retrain = current_time
            logging.info("Model retraining completed successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error retraining model: {e}")
            return False
    
    def save_state(self):
        """Save bot state to file"""
        state = {
            'last_data_update': self.last_data_update,
            'last_model_retrain': self.last_model_retrain,
            'trading_params': self.trading_params,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            with open('bot_state.json', 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving bot state: {e}")
    
    def load_state(self) -> bool:
        """Load bot state from file"""
        try:
            with open('bot_state.json', 'r') as f:
                state = json.load(f)
                self.last_data_update = state['last_data_update']
                self.last_model_retrain = state['last_model_retrain']
                self.trading_params.update(state['trading_params'])
                logging.info("Bot state loaded successfully")
                return True
        except FileNotFoundError:
            logging.info("No previous state found, using default configuration")
            return False
        except Exception as e:
            logging.error(f"Error loading bot state: {e}")
            return False
    
    def run(self):
        """Main bot loop"""
        logging.info("Starting trading bot...")
        self.load_state()
        
        try:
            while True:
                try:
                    # Update data and retrain model if needed
                    if not self.update_data():
                        logging.error("Failed to update data, retrying in 5 minutes...")
                        time.sleep(300)
                        continue
                    
                    if not self.retrain_model():
                        logging.error("Failed to retrain model, retrying in 5 minutes...")
                        time.sleep(300)
                        continue
                    
                    # Initialize trading executor
                    executor = TradingExecutor(
                        symbols=self.symbols,
                        **self.trading_params
                    )
                    
                    # Run trading execution
                    logging.info("Starting trading execution...")
                    executor.execute_trades()
                    
                    # Save current state
                    self.save_state()
                    
                    # Sleep for update interval
                    time.sleep(self.update_interval)
                    
                except Exception as e:
                    logging.error(f"Error in main loop: {e}")
                    time.sleep(300)  # Wait 5 minutes before retrying
                
        except KeyboardInterrupt:
            logging.info("Bot stopped by user")
            self.save_state()
            logging.info("Final account status:")
            self.log_account_info()

def main():
    # Set up logging
    log_file = setup_logging()
    logging.info(f"Logs will be saved to: {log_file}")
    
    # List of symbols to trade
    symbols = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT',
        'ADAUSDT', 'DOGEUSDT', 'XRPUSDT'
    ]
    
    # Initialize and run bot
    try:
        bot = TradingBot(
            symbols=symbols,
            update_interval=3600,      # Update data every hour
            retrain_interval=86400,    # Retrain model daily
            initial_capital=10000.0,
            position_size=0.1,         # 10% of capital per trade
            stop_loss_pct=0.02,        # 2% stop loss
            take_profit_pct=0.06,      # 6% take profit
            max_positions=5
        )
        
        bot.run()
        
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    main() 