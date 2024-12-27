import argparse
from binance_connection import binance_connection
from predictor import make_prediction
from logger_setup import get_logger, log_trade_execution, log_trade_exit, log_critical_error
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Dict, List
import os
from dotenv import load_dotenv
import joblib
from sklearn.ensemble import RandomForestClassifier
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

# Load environment variables
load_dotenv()

# Configure logging
trading_logger = get_logger('trading')

class EnhancedTrader:
    def __init__(self,
                 symbols: List[str],
                 use_real_money: bool = False,
                 max_position_size_usd: float = 1.0,     # Maximum $1 per trade
                 min_confidence: float = 0.75,           # Higher confidence requirement
                 stop_loss_pct: float = 0.015,          # Tight 1.5% stop loss
                 take_profit_pct: float = 0.045,        # 4.5% take profit
                 max_positions: int = 1,                # Only 1 position at a time
                 compound_profits: bool = True):        # Reinvest profits
        
        self.use_real_money = use_real_money
        self.compound_profits = compound_profits
        
        # Performance tracking
        self.performance = {
            'trades': [],
            'daily_pnl': [],
            'metrics': {
                'total_trades': 0,
                'winning_trades': 0,
                'total_pnl': 0.0,
                'best_trade': 0.0,
                'worst_trade': 0.0,
                'avg_profit_per_trade': 0.0,
                'max_drawdown': 0.0
            }
        }
        
        # Create results directory
        os.makedirs('results', exist_ok=True)
        
        # Initialize tracking
        self.active_positions = {}  # symbol -> position info
        self.daily_pnl = 0.0
        self.last_reset = datetime.now().date()
        self.peak_balance = 0.0
        
        # Get initial balance
        self.update_balance()
        self.peak_balance = self.usdt_balance
        
        # Safety warnings
        trading_logger.warning("=" * 50)
        if use_real_money:
            trading_logger.warning("REAL MONEY TRADING - USE EXTREME CAUTION")
        else:
            trading_logger.warning("TESTNET MODE - Paper Trading Only")
        trading_logger.warning(f"Maximum position size: ${max_position_size_usd:.2f}")
        trading_logger.warning(f"Current USDT balance: ${self.usdt_balance:.2f}")
        trading_logger.warning("=" * 50)
    
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
            trading_logger.error(f"Error updating balance: {e}")
            return 0.0
    
    def train_model(self, symbol: str, days: int = 30) -> bool:
        """Train model on recent data"""
        try:
            trading_logger.info(f"Training model for {symbol} using last {days} days of data...")
            
            # Fetch historical data
            klines = binance_connection.get_historical_klines(
                symbol, '1h', days * 24
            )
            
            if not klines:
                trading_logger.error(f"No data available for {symbol}")
                return False
            
            # Prepare DataFrame
            df = pd.DataFrame(klines)
            df['timestamp'] = pd.to_datetime(df['time'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Calculate features
            df = self._calculate_features(df)
            
            # Create labels (1 if price increased in next period)
            df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
            
            # Remove last row (no target) and any NaN values
            df = df.dropna()
            
            # Prepare features and target
            feature_columns = [
                'rsi', 'macd', 'macd_signal', 'macd_diff',
                'bb_upper', 'bb_middle', 'bb_lower',
                'sma_20', 'sma_50', 'ema_12', 'ema_26'
            ]
            
            X = df[feature_columns].values
            y = df['target'].values
            
            # Train model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
            self.model.fit(X, y)
            
            # Save model
            model_path = f'models/real_money_{symbol.lower()}_model.pkl'
            os.makedirs('models', exist_ok=True)
            joblib.dump(self.model, model_path)
            
            trading_logger.info(f"Model trained and saved successfully for {symbol}")
            return True
            
        except Exception as e:
            trading_logger.error(f"Error training model: {e}")
            return False
    
    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for prediction"""
        try:
            # RSI
            rsi = RSIIndicator(df['close'], window=14)
            df['rsi'] = rsi.rsi()
            
            # MACD
            macd = MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
            
            # Bollinger Bands
            bb = BollingerBands(df['close'])
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_lower'] = bb.bollinger_lband()
            
            # Moving Averages
            df['sma_20'] = SMAIndicator(df['close'], window=20).sma_indicator()
            df['sma_50'] = SMAIndicator(df['close'], window=50).sma_indicator()
            df['ema_12'] = EMAIndicator(df['close'], window=12).ema_indicator()
            df['ema_26'] = EMAIndicator(df['close'], window=26).ema_indicator()
            
            return df
            
        except Exception as e:
            trading_logger.error(f"Error calculating features: {e}")
            raise
    
    def update_performance(self, trade_info: Dict):
        """Update performance metrics"""
        try:
            # Update trade history
            self.performance['trades'].append(trade_info)
            
            # Update metrics
            metrics = self.performance['metrics']
            pnl = trade_info.get('pnl', 0.0)
            
            metrics['total_trades'] += 1
            if pnl > 0:
                metrics['winning_trades'] += 1
            
            metrics['total_pnl'] += pnl
            metrics['best_trade'] = max(metrics['best_trade'], pnl)
            metrics['worst_trade'] = min(metrics['worst_trade'], pnl)
            metrics['avg_profit_per_trade'] = metrics['total_pnl'] / metrics['total_trades']
            
            # Update daily PnL
            self.performance['daily_pnl'].append({
                'timestamp': datetime.now(),
                'pnl': pnl,
                'balance': self.usdt_balance
            })
            
            # Update peak balance and drawdown
            if self.usdt_balance > self.peak_balance:
                self.peak_balance = self.usdt_balance
            
            current_drawdown = (self.peak_balance - self.usdt_balance) / self.peak_balance
            metrics['max_drawdown'] = max(metrics['max_drawdown'], current_drawdown)
            
            # Save performance data
            self.save_performance()
            
        except Exception as e:
            trading_logger.error(f"Error updating performance: {e}")
    
    def save_performance(self):
        """Save performance data to file"""
        try:
            # Save trade history
            trades_df = pd.DataFrame(self.performance['trades'])
            trades_df.to_csv('results/trade_history.csv', index=False)
            
            # Save daily PnL
            pnl_df = pd.DataFrame(self.performance['daily_pnl'])
            pnl_df.to_csv('results/daily_pnl.csv', index=False)
            
            # Save metrics summary
            with open('results/performance_metrics.txt', 'w') as f:
                f.write("Performance Metrics Summary\n")
                f.write("=" * 50 + "\n\n")
                
                metrics = self.performance['metrics']
                win_rate = (metrics['winning_trades'] / metrics['total_trades'] * 100) if metrics['total_trades'] > 0 else 0
                
                f.write(f"Total Trades: {metrics['total_trades']}\n")
                f.write(f"Winning Trades: {metrics['winning_trades']}\n")
                f.write(f"Win Rate: {win_rate:.2f}%\n")
                f.write(f"Total PnL: ${metrics['total_pnl']:.2f}\n")
                f.write(f"Best Trade: ${metrics['best_trade']:.2f}\n")
                f.write(f"Worst Trade: ${metrics['worst_trade']:.2f}\n")
                f.write(f"Average Profit per Trade: ${metrics['avg_profit_per_trade']:.2f}\n")
                f.write(f"Maximum Drawdown: {metrics['max_drawdown']*100:.2f}%\n")
                f.write(f"Current Balance: ${self.usdt_balance:.2f}\n")
            
        except Exception as e:
            trading_logger.error(f"Error saving performance data: {e}")
    
    def calculate_position_size(self, symbol: str, prediction: Dict) -> float:
        """Calculate position size with dynamic adjustment"""
        try:
            # Get available balance
            available_balance = self.usdt_balance
            
            # Base position size on available balance
            if self.compound_profits:
                # Use percentage of current balance
                position_size = min(
                    available_balance * 0.3,  # Max 30% of balance
                    self.max_position_size_usd
                )
            else:
                # Use fixed position size
                position_size = min(
                    self.max_position_size_usd,
                    available_balance
                )
            
            # Adjust for volatility
            volatility = self._calculate_volatility(symbol)
            position_size *= (1.0 - volatility)  # Reduce size for high volatility
            
            # Adjust for confidence
            confidence_factor = (prediction['confidence'] - self.min_confidence) / (1 - self.min_confidence)
            position_size *= (0.5 + 0.5 * confidence_factor)  # Scale based on confidence
            
            # Ensure minimum viable position size
            if position_size < 0.5:  # Minimum $0.5 per trade
                return 0.0
            
            return position_size
            
        except Exception as e:
            trading_logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def _calculate_volatility(self, symbol: str) -> float:
        """Calculate recent volatility for position sizing"""
        try:
            klines = binance_connection.get_historical_klines(
                symbol, '1h', 24  # Last 24 hours
            )
            
            if not klines:
                return 0.5  # Default to medium volatility
            
            prices = [float(k['close']) for k in klines]
            returns = np.diff(np.log(prices))
            volatility = np.std(returns)
            
            # Normalize to 0-1 range (assuming max volatility of 0.1)
            normalized_vol = min(volatility / 0.1, 1.0)
            
            return normalized_vol
            
        except Exception as e:
            trading_logger.error(f"Error calculating volatility: {e}")
            return 0.5
    
    def execute_trade(self, symbol: str, prediction: Dict) -> bool:
        """Execute trade with enhanced risk management"""
        try:
            # Additional safety checks
            if len(self.active_positions) >= self.max_positions:
                trading_logger.info("Maximum positions reached")
                return False
            
            if prediction['confidence'] < self.min_confidence:
                trading_logger.info(f"Confidence too low: {prediction['confidence']:.2%}")
                return False
            
            # Calculate position size
            position_size = self.calculate_position_size(symbol, prediction)
            
            if position_size <= 0:
                trading_logger.info("Position size too small")
                return False
            
            # Get current price and calculate levels
            current_price = prediction['current_price']
            
            # Calculate stop loss and take profit with dynamic adjustment
            volatility = self._calculate_volatility(symbol)
            stop_loss_pct = self.stop_loss_pct * (1 + volatility)  # Wider for high volatility
            take_profit_pct = self.take_profit_pct * (1 + volatility)  # Higher for high volatility
            
            stop_loss = current_price * (1 - stop_loss_pct) if prediction['prediction'] == 1 else \
                       current_price * (1 + stop_loss_pct)
            
            take_profit = current_price * (1 + take_profit_pct) if prediction['prediction'] == 1 else \
                         current_price * (1 - take_profit_pct)
            
            # Store position information
            side = 'BUY' if prediction['prediction'] == 1 else 'SELL'
            self.active_positions[symbol] = {
                'side': side,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'position_size': position_size,
                'entry_time': datetime.now(),
                'volatility': volatility
            }
            
            # Log trade execution
            trade_info = {
                'symbol': symbol,
                'side': side,
                'entry_price': current_price,
                'position_size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': prediction['confidence'],
                'volatility': volatility
            }
            log_trade_execution(trading_logger, trade_info)
            
            return True
            
        except Exception as e:
            error_info = {
                'error_type': 'Trade Execution Error',
                'error_message': str(e),
                'symbol': symbol
            }
            log_critical_error(trading_logger, error_info)
            return False
    
    def manage_positions(self):
        """Manage open positions"""
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
                
                # Check stop loss
                if (position['side'] == 'BUY' and current_price <= position['stop_loss']) or \
                   (position['side'] == 'SELL' and current_price >= position['stop_loss']):
                    position['exit_reason'] = 'Stop Loss'
                    self.close_position(symbol)
                
                # Check take profit
                elif (position['side'] == 'BUY' and current_price >= position['take_profit']) or \
                     (position['side'] == 'SELL' and current_price <= position['take_profit']):
                    position['exit_reason'] = 'Take Profit'
                    self.close_position(symbol)
                
                # Update position information
                position['current_price'] = current_price
                position['unrealized_pnl'] = pnl
                
        except Exception as e:
            error_info = {
                'error_type': 'Position Management Error',
                'error_message': str(e)
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
            
            # Log trade exit
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
            self.daily_pnl += pnl
            
        except Exception as e:
            error_info = {
                'error_type': 'Position Close Error',
                'error_message': str(e),
                'symbol': symbol
            }
            log_critical_error(trading_logger, error_info)
    
    def run(self):
        """Run the real money trader"""
        try:
            trading_logger.info("Starting real money trader...")
            
            # Train models first
            for symbol in self.symbols:
                if not self.train_model(symbol):
                    trading_logger.error(f"Failed to train model for {symbol}")
                    continue
            
            trading_logger.info("Models trained successfully")
            
            while True:
                try:
                    # Update balance
                    self.update_balance()
                    
                    # Look for trading opportunities
                    for symbol in self.symbols:
                        if symbol in self.active_positions:
                            continue
                        
                        prediction = make_prediction(symbol)
                        if prediction['success']:
                            self.execute_trade(symbol, prediction)
                    
                    # Manage existing positions
                    self.manage_positions()
                    
                    # Sleep for 1 minute
                    time.sleep(60)
                    
                except KeyboardInterrupt:
                    trading_logger.info("Shutting down real money trader...")
                    break
                except Exception as e:
                    trading_logger.error(f"Error in main loop: {e}")
                    time.sleep(60)  # Wait before retrying
            
        except Exception as e:
            trading_logger.error(f"Fatal error in real money trader: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Enhanced Binance Trading Bot')
    parser.add_argument('--real-money', action='store_true', help='Use real money trading (default: False)')
    parser.add_argument('--train-only', action='store_true', help='Train models only without trading')
    parser.add_argument('--backtest', action='store_true', help='Run backtest before trading')
    parser.add_argument('--max-position', type=float, default=1.0, help='Maximum position size in USD')
    parser.add_argument('--confidence', type=float, default=0.75, help='Minimum prediction confidence')
    parser.add_argument('--compound', action='store_true', help='Enable compound profits')
    args = parser.parse_args()
    
    # Trading pairs (from backtest results)
    symbols = [
        'AMBUSDT',  # Best performer from backtest
        'ARUSDT',   # Second best from backtest
    ]
    
    try:
        if args.backtest:
            # Run backtest first
            trading_logger.info("Running backtest...")
            from backtester import Backtester
            
            backtester = Backtester(
                symbols=symbols,
                initial_capital=3.0,     # Start with $3 USDT
                position_size_pct=0.15,  # Use 15% of capital per trade
                stop_loss_pct=0.01,     # 1% stop loss
                take_profit_pct=0.03,   # 3% take profit
                min_confidence=0.65,    # Minimum 65% confidence
                max_positions=1,        # Maximum 1 position at a time
                days=30                 # Backtest on last 30 days
            )
            
            results = backtester.run_backtest()
            
            # Print backtest results
            print("\nBacktest Results Summary")
            print("=" * 50)
            for metric, value in results['metrics'].items():
                print(f"{metric.replace('_', ' ').title()}: {value:.2f}")
            
            # Ask for confirmation to proceed
            confirm = input("\nDo you want to proceed with live trading? (yes/no): ")
            if confirm.lower() != 'yes':
                print("Trading cancelled.")
                return
        
        if args.real_money:
            # Additional confirmation for real money trading
            confirm = input("\nWARNING: You are about to use REAL MONEY trading. Type 'yes' to confirm: ")
            if confirm.lower() != 'yes':
                print("Real money trading cancelled.")
                return
        
        # Initialize trader
        trader = EnhancedTrader(
            symbols=symbols,
            use_real_money=args.real_money,
            max_position_size_usd=args.max_position,
            min_confidence=args.confidence,
            stop_loss_pct=0.015,           # 1.5% stop loss
            take_profit_pct=0.045,         # 4.5% take profit
            max_positions=1,               # Only 1 position at a time
            compound_profits=args.compound  # Enable/disable compound profits
        )
        
        if args.train_only:
            # Train models and exit
            for symbol in symbols:
                trader.train_model(symbol)
            trading_logger.info("Training completed. Exiting...")
            return
        
        # Print trading configuration
        print("\nTrading Configuration:")
        print("=" * 50)
        print(f"Mode: {'REAL MONEY' if args.real_money else 'TESTNET'}")
        print(f"Max Position Size: ${args.max_position:.2f}")
        print(f"Min Confidence: {args.confidence:.0%}")
        print(f"Compound Profits: {'Enabled' if args.compound else 'Disabled'}")
        print(f"Trading Pairs: {', '.join(symbols)}")
        print("=" * 50)
        
        # Run the trader
        trader.run()
        
    except Exception as e:
        trading_logger.error(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    main() 