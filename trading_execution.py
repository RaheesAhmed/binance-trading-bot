from binance_connection import binance_connection
import pandas as pd
import numpy as np
import joblib
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import os
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Verify we're in testnet mode
if os.getenv('USE_TESTNET', 'true').lower() != 'true':
    raise ValueError("USE_TESTNET must be set to true in .env file")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading.log'),
        logging.StreamHandler()
    ]
)

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('graphs', exist_ok=True)
os.makedirs('logs', exist_ok=True)

class TradingExecutor:
    def __init__(self,
                 symbols: List[str],
                 initial_capital: float = 10000.0,
                 position_size: float = 0.1,  # 10% of capital per trade
                 stop_loss_pct: float = 0.02,  # 2% stop loss
                 take_profit_pct: float = 0.06,  # 6% take profit
                 max_positions: int = 5):  # Maximum number of simultaneous positions
        
        # Double-check testnet configuration
        if not binance_connection.use_testnet:
            logging.error("Binance connection is not in testnet mode!")
            raise ValueError("Binance connection must be configured for testnet!")
            
        logging.warning("Running in TESTNET mode - Using paper trading only!")
        
        self.symbols = self.validate_symbols(symbols)  # Validate symbols first
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_positions = max_positions
        
        # Load model and related files
        try:
            self.model = joblib.load('models/trading_model.pkl')
            self.scaler = joblib.load('models/feature_scaler.pkl')
            with open('data/feature_names.json', 'r') as f:
                self.feature_names = json.load(f)['features']
        except Exception as e:
            logging.error(f"Error loading model files: {e}")
            raise
        
        # Initialize positions tracking
        self.active_positions = {}  # symbol -> position_info
        
        # Get account balance
        self.update_account_balance()
        logging.info(f"Initial USDT balance (TESTNET): {self.usdt_balance}")
        
        # Add safety warning
        logging.warning("=" * 50)
        logging.warning("TESTNET MODE ACTIVE - NO REAL MONEY WILL BE USED")
        logging.warning("All trades are simulated with test funds")
        logging.warning("=" * 50)

    def validate_symbols(self, symbols: List[str]) -> List[str]:
        """Validate that symbols are available for trading"""
        valid_symbols = []
        exchange_info = binance_connection.client.get_exchange_info()
        valid_trading_pairs = {s['symbol'] for s in exchange_info['symbols'] if s['status'] == 'TRADING'}
        
        for symbol in symbols:
            if symbol in valid_trading_pairs:
                valid_symbols.append(symbol)
            else:
                logging.warning(f"Symbol {symbol} is not available for trading, skipping.")
        
        return valid_symbols

    def update_account_balance(self):
        """Update the current USDT balance"""
        try:
            account = binance_connection.client.get_account()
            usdt_balance = next(
                (float(asset['free']) for asset in account['balances'] if asset['asset'] == 'USDT'),
                0.0
            )
            self.usdt_balance = usdt_balance
            return usdt_balance
        except Exception as e:
            logging.error(f"Error getting account balance: {e}")
            self.usdt_balance = 0.0
            return 0.0

    def get_features(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get and prepare features for prediction"""
        try:
            # Get latest candle data
            klines = binance_connection.get_historical_klines(symbol, '1h', 200)
            if not klines:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(klines)
            df['timestamp'] = pd.to_datetime([k['time'] for k in klines], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Prepare features
            feature_data = pd.DataFrame()
            feature_data['open'] = pd.to_numeric([k['open'] for k in klines])
            feature_data['high'] = pd.to_numeric([k['high'] for k in klines])
            feature_data['low'] = pd.to_numeric([k['low'] for k in klines])
            feature_data['close'] = pd.to_numeric([k['close'] for k in klines])
            feature_data['volume'] = pd.to_numeric([k['volume'] for k in klines])
            
            # Add technical indicators
            # RSI
            feature_data['rsi'] = self.calculate_rsi(feature_data['close'])
            
            # MACD
            macd_data = self.calculate_macd(feature_data['close'])
            feature_data['macd'] = macd_data['macd']
            feature_data['macd_signal'] = macd_data['signal']
            feature_data['macd_diff'] = macd_data['diff']
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            feature_data['bb_mid'] = feature_data['close'].rolling(window=bb_period).mean()
            bb_std_dev = feature_data['close'].rolling(window=bb_period).std()
            feature_data['bb_high'] = feature_data['bb_mid'] + (bb_std * bb_std_dev)
            feature_data['bb_low'] = feature_data['bb_mid'] - (bb_std * bb_std_dev)
            
            # Simple Moving Averages
            feature_data['sma_20'] = feature_data['close'].rolling(window=20).mean()
            feature_data['sma_50'] = feature_data['close'].rolling(window=50).mean()
            feature_data['sma_200'] = feature_data['close'].rolling(window=200).mean()
            
            # Average True Range (ATR)
            high_low = feature_data['high'] - feature_data['low']
            high_close = (feature_data['high'] - feature_data['close'].shift()).abs()
            low_close = (feature_data['low'] - feature_data['close'].shift()).abs()
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            feature_data['atr'] = true_range.rolling(14).mean()
            
            # Price and Volume Momentum
            feature_data['price_momentum'] = feature_data['close'].pct_change()
            feature_data['volume_sma'] = feature_data['volume'].rolling(window=20).mean()
            feature_data['volume_momentum'] = feature_data['volume'].pct_change()
            
            # Handle any NaN values using newer method
            feature_data = feature_data.ffill().bfill()
            
            # Return only the latest values for prediction
            return feature_data.iloc[-1:][self.feature_names]
            
        except Exception as e:
            logging.error(f"Error getting features for {symbol}: {str(e)}")
            return None

    def check_balance_for_trade(self, symbol: str, quantity: float, side: str) -> bool:
        """Check if we have enough balance for the trade"""
        try:
            # Update current balance
            self.update_account_balance()
            
            # Get symbol price
            price = binance_connection.get_symbol_price(symbol)
            if not price:
                return False
                
            # Calculate required balance
            required_usdt = price * quantity
            
            # Add 1% buffer for fees and price movements
            required_usdt *= 1.01
            
            # For selling, check if we have the asset
            if side == 'SELL':
                asset = symbol.replace('USDT', '')
                account = binance_connection.client.get_account()
                asset_balance = next(
                    (float(b['free']) for b in account['balances'] if b['asset'] == asset),
                    0.0
                )
                return asset_balance >= quantity
            
            # For buying, check USDT balance
            return self.usdt_balance >= required_usdt
            
        except Exception as e:
            logging.error(f"Error checking balance: {e}")
            return False

    def place_order(self, symbol: str, side: str, quantity: float, stop_loss: float, take_profit: float) -> bool:
        """Place an order with stop loss and take profit"""
        try:
            # Check balance before placing order
            if not self.check_balance_for_trade(symbol, quantity, side):
                logging.warning(f"Insufficient balance for {side} order of {quantity} {symbol}")
                return False
            
            # Place the main order
            order = binance_connection.place_order(
                symbol=symbol,
                side=side,
                order_type='MARKET',
                quantity=quantity
            )
            
            if not order:
                return False
            
            # Get the filled price
            filled_price = float(order['fills'][0]['price'])
            logging.info(f"Main order filled at price: {filled_price}")
            
            # Place stop loss order
            stop_loss_order = binance_connection.place_order(
                symbol=symbol,
                side='SELL' if side == 'BUY' else 'BUY',
                order_type='STOP_LOSS',
                quantity=quantity,
                price=stop_loss,
                stopPrice=stop_loss
            )
            
            # Place take profit order
            take_profit_order = binance_connection.place_order(
                symbol=symbol,
                side='SELL' if side == 'BUY' else 'BUY',
                order_type='LIMIT',
                quantity=quantity,
                price=take_profit
            )
            
            # Track the position
            self.active_positions[symbol] = {
                'side': side,
                'quantity': quantity,
                'entry_price': filled_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'entry_time': datetime.now(),
                'orders': {
                    'main': order['orderId'],
                    'stop_loss': stop_loss_order['orderId'] if stop_loss_order else None,
                    'take_profit': take_profit_order['orderId'] if take_profit_order else None
                }
            }
            
            logging.info(f"Opened {side} position for {symbol} at {filled_price}")
            logging.info(f"Stop Loss: {stop_loss}, Take Profit: {take_profit}")
            return True
            
        except Exception as e:
            logging.error(f"Error placing orders for {symbol}: {str(e)}")
            return False

    def calculate_position_size(self, symbol: str) -> float:
        """Calculate position size based on available capital and risk"""
        try:
            # Update current balance
            self.update_account_balance()
            
            # Get symbol price
            price = binance_connection.get_symbol_price(symbol)
            if not price:
                return 0.0
            
            # Calculate position size based on current balance and price
            # Use only 1% of available balance per trade for safety
            position_value = self.usdt_balance * 0.01  # Reduced from 0.1 (10%) to 0.01 (1%)
            quantity = position_value / price
            
            # Get symbol info for quantity precision
            info = binance_connection.client.get_symbol_info(symbol)
            lot_size_filter = next(filter(lambda x: x['filterType'] == 'LOT_SIZE', info['filters']))
            step_size = float(lot_size_filter['stepSize'])
            
            # Calculate decimal places for rounding
            decimal_places = len(str(step_size).split('.')[-1].rstrip('0'))
            
            # Round down to avoid exceeding available balance
            quantity = float(format(quantity, f'.{decimal_places}f'))
            
            logging.info(f"Calculated position size for {symbol}: {quantity} at price {price}")
            return quantity
            
        except Exception as e:
            logging.error(f"Error calculating position size for {symbol}: {str(e)}")
            return 0.0

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """Calculate MACD indicator"""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return {
            'macd': macd,
            'signal': signal,
            'diff': macd - signal
        }
    
    def close_position(self, symbol: str) -> bool:
        """Close an open position"""
        try:
            if symbol not in self.active_positions:
                return False
            
            position = self.active_positions[symbol]
            
            # Cancel existing stop loss and take profit orders
            for order_type in ['stop_loss', 'take_profit']:
                try:
                    binance_connection.client.cancel_order(
                        symbol=symbol,
                        orderId=position['orders'][order_type]
                    )
                except:
                    pass
            
            # Place closing market order
            close_side = 'SELL' if position['side'] == 'BUY' else 'BUY'
            order = binance_connection.place_order(
                symbol=symbol,
                side=close_side,
                order_type='MARKET',
                quantity=position['quantity']
            )
            
            if order:
                logging.info(f"Closed position for {symbol}")
                del self.active_positions[symbol]
                return True
                
            return False
            
        except Exception as e:
            logging.error(f"Error closing position for {symbol}: {str(e)}")
            return False
    
    def execute_trades(self):
        """Main trading loop"""
        while True:
            try:
                # Check each symbol
                for symbol in self.symbols:
                    # Skip if we already have a position
                    if symbol in self.active_positions:
                        continue
                    
                    # Skip if we have reached max positions
                    if len(self.active_positions) >= self.max_positions:
                        continue
                    
                    # Get features and make prediction
                    features = self.get_features(symbol)
                    if features is None:
                        continue
                    
                    X = self.scaler.transform(features)
                    prediction = self.model.predict(X)[0]
                    
                    # Get current price
                    current_price = binance_connection.get_symbol_price(symbol)
                    if not current_price:
                        continue
                    
                    # Calculate position size
                    quantity = self.calculate_position_size(symbol)
                    if quantity <= 0:
                        continue
                    
                    # Place orders based on prediction
                    if prediction == 1:  # Long position
                        stop_loss = current_price * (1 - self.stop_loss_pct)
                        take_profit = current_price * (1 + self.take_profit_pct)
                        self.place_order(symbol, 'BUY', quantity, stop_loss, take_profit)
                        
                    else:  # Short position
                        stop_loss = current_price * (1 + self.stop_loss_pct)
                        take_profit = current_price * (1 - self.take_profit_pct)
                        self.place_order(symbol, 'SELL', quantity, stop_loss, take_profit)
                
                # Check existing positions
                for symbol in list(self.active_positions.keys()):
                    position = self.active_positions[symbol]
                    
                    # Close positions older than 24 hours
                    if datetime.now() - position['entry_time'] > timedelta(hours=24):
                        self.close_position(symbol)
                
                # Sleep to avoid API rate limits
                time.sleep(5)
                
            except Exception as e:
                logging.error(f"Error in trading loop: {str(e)}")
                time.sleep(10)

def main():
    # List of symbols to trade (will be validated)
    # Removed ATAUSDT and BBUSDT due to poor performance
    # Added top performing pairs based on backtest results
    symbols = [
        'AMBUSDT',  # Best performer: 105.58% return, 6.60 Sharpe, 82.35% win rate
        'ARUSDT',   # Second best: 98.60% return, 6.78 Sharpe, 80.34% win rate
        'AIUSDT',   # Third best: 97.06% return, 5.98 Sharpe, 74.70% win rate
        'ADXUSDT',  # Solid performer: 62.77% return, 5.31 Sharpe, 78.47% win rate
        'ACAUSDT',  # Good performer: 75.47% return, 2.77 Sharpe, 72.32% win rate
    ]
    
    try:
        # Initialize and run trading executor
        executor = TradingExecutor(
            symbols=symbols,
            initial_capital=10000.0,  # Using 10k USDT test money
            position_size=0.1,
            stop_loss_pct=0.02,
            take_profit_pct=0.06,
            max_positions=5
        )
        
        logging.info("Starting trading execution in TESTNET mode...")
        executor.execute_trades()
        
    except Exception as e:
        logging.error(f"Fatal error in trading execution: {e}")
        raise

if __name__ == "__main__":
    main() 