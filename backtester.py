import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
from logger_setup import get_logger
from binance_connection import binance_connection
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from pathlib import Path
import ta
import traceback
import json

# Configure logging
backtest_logger = get_logger('backtest')

# Try to import seaborn for better plots, but don't require it
try:
    import seaborn as sns
    sns.set_style('darkgrid')
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    backtest_logger.warning("Seaborn not installed. Using basic matplotlib styling.")
    plt.style.use('seaborn-darkgrid' if 'seaborn-darkgrid' in plt.style.available else 'default')

class Backtester:
    def __init__(self,
                 symbols: List[str],
                 initial_capital: float = 3.0,    # Start with $3 USDT
                 position_size_pct: float = 0.15,  # Reduced from 30% to 15% per trade
                 stop_loss_pct: float = 0.01,     # Tighter 1% stop loss
                 take_profit_pct: float = 0.03,   # Lower 3% take profit for more frequent wins
                 min_confidence: float = 0.65,    # Minimum prediction confidence
                 max_positions: int = 1,          # Maximum simultaneous positions
                 days: int = 30):                 # Backtest period
        
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.min_confidence = min_confidence
        self.max_positions = max_positions
        self.days = days
        
        # Performance tracking
        self.results = {
            'trades': [],
            'daily_pnl': [],
            'metrics': {},
            'active_positions': 0
        }
        
        # Create directories for results
        Path('results').mkdir(exist_ok=True)
        Path('results/plots').mkdir(exist_ok=True)
        
        backtest_logger.info(
            f"Initialized backtester with:"
            f"\n- Initial capital: ${initial_capital:.2f}"
            f"\n- Position size: {position_size_pct:.1%}"
            f"\n- Stop loss: {stop_loss_pct:.1%}"
            f"\n- Take profit: {take_profit_pct:.1%}"
            f"\n- Min confidence: {min_confidence:.1%}"
            f"\n- Max positions: {max_positions}"
            f"\n- Test period: {days} days"
        )
    
    def run_backtest(self) -> Dict:
        """Run backtest on historical data"""
        try:
            total_pnl = 0
            capital = self.initial_capital
            win_trades = 0
            total_trades = 0
            max_capital = capital
            trades = []
            daily_pnl = []
            
            for symbol in self.symbols:
                backtest_logger.info(f"Running backtest for {symbol}...")
                
                # Get historical data
                klines = binance_connection.get_historical_klines(
                    symbol, '1h', self.days * 24
                )
                
                if not klines:
                    backtest_logger.error(f"No data available for {symbol}")
                    continue
                
                # Prepare DataFrame
                df = pd.DataFrame(klines)
                df['timestamp'] = pd.to_datetime(df['time'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Calculate features and make predictions
                df, feature_columns = self._calculate_features(df)
                predictions, confidences = self._make_predictions(df, symbol)
                
                # Skip if no valid predictions
                if len(predictions) == 0 or len(confidences) == 0:
                    backtest_logger.warning(f"No valid predictions for {symbol}, skipping...")
                    continue
                
                # Ensure predictions array has same length as df
                df = df.iloc[:len(predictions)]  # Trim df to match predictions length
                
                # Simulate trading
                position = None
                entry_price = 0
                position_size = 0
                entry_time = None
                
                for i in range(len(df) - 1):  # -1 to ensure we have next price
                    current_time = df.index[i]
                    current_price = float(df['close'].iloc[i])
                    next_price = float(df['close'].iloc[i + 1])
                    prediction = predictions[i]
                    confidence = confidences[i]
                    
                    # Check if we can open a position
                    if position is None:
                        if confidence >= self.min_confidence:
                            # Calculate position size based on ATR
                            position_size = self._calculate_position_size(
                                capital, current_price, float(df['atr'].iloc[i])
                            )
                            
                            if position_size > 0:
                                if prediction == 1:  # Buy signal
                                    position = 'long'
                                    entry_price = current_price
                                    entry_time = current_time
                                    
                                    trades.append({
                                        'symbol': symbol,
                                        'entry_time': entry_time,
                                        'entry_price': entry_price,
                                        'position_size': position_size,
                                        'type': 'long',
                                        'confidence': confidence
                                    })
                                    backtest_logger.info(
                                        f"Opening LONG position for {symbol} at {entry_price:.8f}"
                                        f"\n- Position size: ${position_size:.2f}"
                                        f"\n- Confidence: {confidence:.2%}"
                                    )
                                
                                elif prediction == 0:  # Sell signal
                                    position = 'short'
                                    entry_price = current_price
                                    entry_time = current_time
                                    
                                    trades.append({
                                        'symbol': symbol,
                                        'entry_time': entry_time,
                                        'entry_price': entry_price,
                                        'position_size': position_size,
                                        'type': 'short',
                                        'confidence': confidence
                                    })
                                    backtest_logger.info(
                                        f"Opening SHORT position for {symbol} at {entry_price:.8f}"
                                        f"\n- Position size: ${position_size:.2f}"
                                        f"\n- Confidence: {confidence:.2%}"
                                    )
                    
                    # Check if we need to close position
                    elif position is not None:
                        pnl = 0
                        exit_reason = None
                        
                        # Calculate price movement
                        price_change = (next_price - entry_price) / entry_price
                        
                        if position == 'long':
                            # Check stop loss
                            if price_change <= -self.stop_loss_pct:
                                pnl = -position_size * self.stop_loss_pct
                                exit_reason = 'stop_loss'
                            # Check take profit
                            elif price_change >= self.take_profit_pct:
                                pnl = position_size * self.take_profit_pct
                                exit_reason = 'take_profit'
                            # Check trend reversal
                            elif prediction == 0 and confidence >= self.min_confidence:
                                pnl = position_size * price_change
                                exit_reason = 'signal_reversal'
                        
                        else:  # Short position
                            # Check stop loss
                            if price_change >= self.stop_loss_pct:
                                pnl = -position_size * self.stop_loss_pct
                                exit_reason = 'stop_loss'
                            # Check take profit
                            elif price_change <= -self.take_profit_pct:
                                pnl = position_size * self.take_profit_pct
                                exit_reason = 'take_profit'
                            # Check trend reversal
                            elif prediction == 1 and confidence >= self.min_confidence:
                                pnl = position_size * -price_change
                                exit_reason = 'signal_reversal'
                        
                        # Close position if exit condition met
                        if exit_reason:
                            capital += pnl
                            total_pnl += pnl
                            total_trades += 1
                            if pnl > 0:
                                win_trades += 1
                            
                            # Update max capital for drawdown calculation
                            max_capital = max(max_capital, capital)
                            
                            # Record trade exit
                            trades[-1].update({
                                'exit_time': current_time,
                                'exit_price': next_price,
                                'pnl': pnl,
                                'exit_reason': exit_reason
                            })
                            
                            backtest_logger.info(
                                f"Closing {position.upper()} position for {symbol}"
                                f"\n- Entry price: {entry_price:.8f}"
                                f"\n- Exit price: {next_price:.8f}"
                                f"\n- PnL: ${pnl:.2f}"
                                f"\n- Exit reason: {exit_reason}"
                            )
                            
                            # Record daily PnL
                            daily_pnl.append({
                                'timestamp': current_time,
                                'pnl': pnl,
                                'capital': capital
                            })
                            
                            position = None
                            entry_price = 0
                            position_size = 0
                            entry_time = None
            
            # Calculate final metrics
            win_rate = win_trades / total_trades if total_trades > 0 else 0
            total_return = (capital - self.initial_capital) / self.initial_capital
            max_drawdown = (max_capital - min([p['capital'] for p in daily_pnl], default=capital)) / max_capital if daily_pnl else 0
            
            # Store results
            self.results = {
                'trades': trades,
                'daily_pnl': daily_pnl,
                'metrics': {
                    'total_pnl': total_pnl,
                    'total_return_pct': total_return * 100,
                    'win_rate': win_rate * 100,
                    'total_trades': total_trades,
                    'win_trades': win_trades,
                    'loss_trades': total_trades - win_trades,
                    'final_capital': capital,
                    'max_drawdown_pct': max_drawdown * 100
                }
            }
            
            # Generate performance plots
            self._generate_plots()
            
            # Save results
            self._save_results()
            
            return self.results
            
        except Exception as e:
            backtest_logger.error(f"Error in backtest: {e}")
            raise
    
    def _calculate_features(self, df):
        """Calculate technical indicators"""
        try:
            # RSI with multiple periods
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            df['rsi_slow'] = ta.momentum.RSIIndicator(df['close'], window=21).rsi()
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'])
            df['bb_high'] = bb.bollinger_hband()
            df['bb_low'] = bb.bollinger_lband()
            df['bb_mid'] = bb.bollinger_mavg()
            df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
            
            # Moving Averages
            df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
            df['sma_50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
            df['sma_200'] = ta.trend.SMAIndicator(df['close'], window=200).sma_indicator()
            df['ema_9'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
            
            # ATR and Volatility
            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
            df['volatility'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean()
            
            # Momentum Indicators
            df['price_momentum'] = df['close'].pct_change(periods=10)
            df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()
            df['stoch_k'] = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close']).stoch()
            df['stoch_d'] = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close']).stoch_signal()
            
            # Volume Indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_momentum'] = df['volume'].pct_change(periods=10)
            df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
            
            # Ensure all required features are present
            feature_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'rsi', 'rsi_slow', 'macd', 'macd_signal', 'macd_diff',
                'bb_high', 'bb_low', 'bb_mid', 'bb_width',
                'sma_20', 'sma_50', 'sma_200', 'ema_9',
                'atr', 'volatility', 'price_momentum',
                'cci', 'stoch_k', 'stoch_d',
                'volume_sma', 'volume_momentum', 'obv'
            ]
            
            # Handle NaN values
            df = df.ffill().bfill()
            
            return df, feature_columns
            
        except Exception as e:
            backtest_logger.error(f"Error calculating features: {e}")
            raise
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def _make_predictions(self, df, symbol):
        """Make predictions using the trained model"""
        try:
            # Load model and scaler
            model = joblib.load('models/trading_model.pkl')
            scaler = joblib.load('models/feature_scaler.pkl')
            
            # Load feature names
            with open('data/feature_names.json', 'r') as f:
                feature_names = json.load(f)['features']
            
            # Drop rows with NaN values
            df_clean = df.dropna()
            if len(df_clean) == 0:
                backtest_logger.warning("No valid data after dropping NaN values")
                return np.array([]), np.array([])
            
            backtest_logger.info(f"Processing {len(df_clean)} rows for {symbol}")
            
            # Create feature matrix using saved feature names
            try:
                X = df_clean[feature_names].values
            except KeyError as e:
                backtest_logger.error(f"Missing features for {symbol}: {e}")
                backtest_logger.error(f"Available features: {df_clean.columns.tolist()}")
                return np.array([]), np.array([])
            
            # Scale features
            X_scaled = scaler.transform(X)
            
            # Make predictions
            predictions = model.predict(X_scaled)
            confidences = np.max(model.predict_proba(X_scaled), axis=1)
            
            # Log prediction distribution
            unique, counts = np.unique(predictions, return_counts=True)
            distribution = dict(zip(unique, counts))
            backtest_logger.info(f"Prediction distribution for {symbol}: {distribution}")
            
            # Ensure predictions array matches data length
            if len(predictions) != len(df_clean):
                backtest_logger.error(f"Prediction length mismatch: {len(predictions)} vs {len(df_clean)}")
                return np.array([]), np.array([])
            
            return predictions, confidences
            
        except Exception as e:
            backtest_logger.error(f"Error making predictions: {e}")
            traceback.print_exc()
            return np.array([]), np.array([])
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from daily PnL"""
        if not self.results['daily_pnl']:
            return 0.0
        
        df = pd.DataFrame(self.results['daily_pnl'])
        df['cumulative_pnl'] = df['pnl'].cumsum()
        df['peak'] = df['cumulative_pnl'].cummax()
        df['drawdown'] = (df['peak'] - df['cumulative_pnl']) / df['peak']
        
        return df['drawdown'].max()
    
    def _generate_plots(self):
        """Generate performance visualization plots"""
        try:
            if not self.results['trades'] or not self.results['daily_pnl']:
                backtest_logger.warning("No trades or PnL data to plot")
                return
            
            # Create DataFrames from results
            trades_df = pd.DataFrame(self.results['trades'])
            pnl_df = pd.DataFrame(self.results['daily_pnl'])
            
            # Set figure style
            plt.rcParams['figure.figsize'] = [12, 6]
            plt.rcParams['figure.dpi'] = 100
            
            # Plot 1: Cumulative PnL
            plt.figure()
            pnl_df['cumulative_pnl'] = pnl_df['pnl'].cumsum()
            plt.plot(pnl_df['timestamp'], pnl_df['cumulative_pnl'], linewidth=2)
            plt.title('Cumulative PnL Over Time', fontsize=12, pad=15)
            plt.xlabel('Date', fontsize=10)
            plt.ylabel('PnL (USDT)', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('results/plots/cumulative_pnl.png')
            plt.close()
            
            # Plot 2: Trade Distribution
            plt.figure()
            plt.hist(trades_df['pnl'].dropna(), bins=30, edgecolor='black')
            plt.title('Trade PnL Distribution', fontsize=12, pad=15)
            plt.xlabel('PnL (USDT)', fontsize=10)
            plt.ylabel('Frequency', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('results/plots/trade_distribution.png')
            plt.close()
            
            # Plot 3: Capital Growth
            plt.figure()
            plt.plot(pnl_df['timestamp'], pnl_df['capital'], linewidth=2)
            plt.title('Capital Growth Over Time', fontsize=12, pad=15)
            plt.xlabel('Date', fontsize=10)
            plt.ylabel('Capital (USDT)', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('results/plots/capital_growth.png')
            plt.close()
            
            # Plot 4: Win/Loss Ratio
            plt.figure()
            win_trades = trades_df[trades_df['pnl'] > 0]
            loss_trades = trades_df[trades_df['pnl'] <= 0]
            plt.bar(['Winning Trades', 'Losing Trades'], 
                   [len(win_trades), len(loss_trades)],
                   color=['green', 'red'])
            plt.title('Win/Loss Distribution', fontsize=12, pad=15)
            plt.ylabel('Number of Trades', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('results/plots/win_loss_ratio.png')
            plt.close()
            
            backtest_logger.info("Generated performance plots successfully")
            
        except Exception as e:
            backtest_logger.error(f"Error generating plots: {e}", exc_info=True)
            backtest_logger.warning("Continuing without plots...")
    
    def _save_results(self):
        """Save backtest results to file"""
        try:
            if not self.results['trades'] or not self.results['daily_pnl']:
                backtest_logger.warning("No trades or PnL data to save")
                return
            
            # Save detailed results
            trades_df = pd.DataFrame(self.results['trades'])
            trades_df.to_csv('results/backtest_trades.csv', index=False)
            
            # Save daily PnL
            pnl_df = pd.DataFrame(self.results['daily_pnl'])
            pnl_df.to_csv('results/daily_pnl.csv', index=False)
            
            # Calculate additional metrics
            win_trades = trades_df[trades_df['pnl'] > 0]
            loss_trades = trades_df[trades_df['pnl'] <= 0]
            
            avg_win = win_trades['pnl'].mean() if len(win_trades) > 0 else 0
            avg_loss = loss_trades['pnl'].mean() if len(loss_trades) > 0 else 0
            profit_factor = abs(win_trades['pnl'].sum() / loss_trades['pnl'].sum()) if len(loss_trades) > 0 and loss_trades['pnl'].sum() != 0 else float('inf')
            
            # Save metrics summary
            with open('results/backtest_metrics.txt', 'w') as f:
                f.write("Backtest Results Summary\n")
                f.write("=" * 50 + "\n\n")
                
                metrics = self.results['metrics']
                f.write(f"Initial Capital: ${self.initial_capital:.2f}\n")
                f.write(f"Final Capital: ${metrics['final_capital']:.2f}\n")
                f.write(f"Total Return: {metrics['total_return_pct']:.2f}%\n")
                f.write(f"Total PnL: ${metrics['total_pnl']:.2f}\n\n")
                
                f.write(f"Total Trades: {metrics['total_trades']}\n")
                f.write(f"Winning Trades: {metrics['win_trades']}\n")
                f.write(f"Losing Trades: {metrics['loss_trades']}\n")
                f.write(f"Win Rate: {metrics['win_rate']:.2f}%\n\n")
                
                f.write(f"Average Winning Trade: ${avg_win:.2f}\n")
                f.write(f"Average Losing Trade: ${avg_loss:.2f}\n")
                f.write(f"Profit Factor: {profit_factor:.2f}\n")
                f.write(f"Maximum Drawdown: {metrics['max_drawdown_pct']:.2f}%\n")
            
            backtest_logger.info("Saved backtest results to 'results' directory")
            
        except Exception as e:
            backtest_logger.error(f"Error saving results: {e}", exc_info=True)
    
    def _calculate_position_size(self, capital: float, current_price: float, atr: float) -> float:
        """Calculate position size based on capital and volatility"""
        try:
            # Base position size (15% of capital)
            base_size = capital * self.position_size_pct
            
            # Less aggressive volatility adjustment
            volatility_factor = 1.0 - min(atr / current_price, 0.3)  # Cap at 30% reduction
            adjusted_size = base_size * volatility_factor
            
            # Lower minimum position size for small capital
            min_position = min(0.1, capital * 0.05)  # Min of $0.1 or 5% of capital
            if adjusted_size < min_position:
                backtest_logger.debug(
                    f"Position size too small: ${adjusted_size:.2f} < ${min_position:.2f}"
                    f"\n- Capital: ${capital:.2f}"
                    f"\n- Base size: ${base_size:.2f}"
                    f"\n- Volatility factor: {volatility_factor:.2f}"
                )
                return 0.0
            
            backtest_logger.debug(
                f"Calculated position size: ${adjusted_size:.2f}"
                f"\n- Capital: ${capital:.2f}"
                f"\n- Base size: ${base_size:.2f}"
                f"\n- Volatility factor: {volatility_factor:.2f}"
            )
            
            return adjusted_size
            
        except Exception as e:
            backtest_logger.error(f"Error calculating position size: {e}")
            return 0.0

def main():
    # Example usage
    symbols = [
        'AMBUSDT',  # Best performer
        'ARUSDT',   # Second best
    ]
    
    backtester = Backtester(
        symbols=symbols,
        initial_capital=3.0,     # Start with $3 USDT
        position_size_pct=0.15,   # Reduced from 30% to 15% per trade
        stop_loss_pct=0.01,     # Tighter 1% stop loss
        take_profit_pct=0.03,   # Lower 3% take profit for more frequent wins
        min_confidence=0.65,    # Minimum prediction confidence
        max_positions=1,          # Maximum simultaneous positions
        days=30                  # Backtest on last 30 days
    )
    
    results = backtester.run_backtest()
    
    # Print summary
    print("\nBacktest Results Summary")
    print("=" * 50)
    for metric, value in results['metrics'].items():
        print(f"{metric.replace('_', ' ').title()}: {value:.2f}")

if __name__ == "__main__":
    main() 