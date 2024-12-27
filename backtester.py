import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime, timedelta
from logger_setup import get_logger
from binance_connection import binance_connection
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from pathlib import Path

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
                 position_size_pct: float = 0.3,  # Use 30% of capital per trade
                 stop_loss_pct: float = 0.015,    # 1.5% stop loss
                 take_profit_pct: float = 0.045,  # 4.5% take profit
                 days: int = 30):                 # Backtest period
        
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.days = days
        
        # Performance tracking
        self.results = {
            'trades': [],
            'daily_pnl': [],
            'metrics': {}
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
            f"\n- Test period: {days} days"
        )
    
    def run_backtest(self) -> Dict:
        """Run backtest on historical data"""
        try:
            total_pnl = 0
            capital = self.initial_capital
            win_trades = 0
            total_trades = 0
            
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
                df = self._calculate_features(df)
                predictions = self._make_predictions(df, symbol)
                
                # Simulate trading
                position = None
                entry_price = 0
                position_size = 0
                
                for i in range(len(df) - 1):
                    current_price = float(df['close'].iloc[i])
                    next_price = float(df['close'].iloc[i + 1])
                    prediction = predictions[i]
                    timestamp = df.index[i]
                    
                    # Check if we can open a position
                    if position is None and capital > 0:
                        # Calculate position size (30% of current capital)
                        position_size = capital * self.position_size_pct
                        
                        if prediction == 1:  # Buy signal
                            position = 'long'
                            entry_price = current_price
                            
                            self.results['trades'].append({
                                'symbol': symbol,
                                'timestamp': timestamp,
                                'type': 'entry',
                                'side': 'buy',
                                'price': entry_price,
                                'size': position_size
                            })
                        
                        elif prediction == 0:  # Sell signal
                            position = 'short'
                            entry_price = current_price
                            
                            self.results['trades'].append({
                                'symbol': symbol,
                                'timestamp': timestamp,
                                'type': 'entry',
                                'side': 'sell',
                                'price': entry_price,
                                'size': position_size
                            })
                    
                    # Check if we need to close position
                    elif position is not None:
                        pnl = 0
                        exit_reason = None
                        
                        # Calculate price movement
                        price_change = (next_price - entry_price) / entry_price
                        
                        if position == 'long':
                            # Check stop loss
                            if price_change <= -self.stop_loss_pct:
                                pnl = position_size * -self.stop_loss_pct
                                exit_reason = 'stop_loss'
                            # Check take profit
                            elif price_change >= self.take_profit_pct:
                                pnl = position_size * self.take_profit_pct
                                exit_reason = 'take_profit'
                        
                        else:  # Short position
                            # Check stop loss
                            if price_change >= self.stop_loss_pct:
                                pnl = position_size * -self.stop_loss_pct
                                exit_reason = 'stop_loss'
                            # Check take profit
                            elif price_change <= -self.take_profit_pct:
                                pnl = position_size * self.take_profit_pct
                                exit_reason = 'take_profit'
                        
                        # Close position if stop loss or take profit hit
                        if exit_reason:
                            capital += pnl
                            total_pnl += pnl
                            total_trades += 1
                            if pnl > 0:
                                win_trades += 1
                            
                            self.results['trades'].append({
                                'symbol': symbol,
                                'timestamp': timestamp,
                                'type': 'exit',
                                'side': 'sell' if position == 'long' else 'buy',
                                'price': next_price,
                                'size': position_size,
                                'pnl': pnl,
                                'exit_reason': exit_reason
                            })
                            
                            position = None
                            self.results['daily_pnl'].append({
                                'timestamp': timestamp,
                                'pnl': pnl,
                                'capital': capital
                            })
            
            # Calculate final metrics
            win_rate = win_trades / total_trades if total_trades > 0 else 0
            total_return = (capital - self.initial_capital) / self.initial_capital
            
            self.results['metrics'] = {
                'total_pnl': total_pnl,
                'total_return_pct': total_return * 100,
                'win_rate': win_rate * 100,
                'total_trades': total_trades,
                'win_trades': win_trades,
                'loss_trades': total_trades - win_trades,
                'final_capital': capital,
                'max_drawdown_pct': self._calculate_max_drawdown() * 100
            }
            
            # Generate performance plots
            self._generate_plots()
            
            # Save results
            self._save_results()
            
            return self.results
            
        except Exception as e:
            backtest_logger.error(f"Error in backtest: {e}")
            raise
    
    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        try:
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_diff'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['bb_upper'] = df['sma_20'] + 2 * df['close'].rolling(window=20).std()
            df['bb_lower'] = df['sma_20'] - 2 * df['close'].rolling(window=20).std()
            
            # Volatility
            df['atr'] = self._calculate_atr(df)
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            return df
            
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
    
    def _make_predictions(self, df: pd.DataFrame, symbol: str) -> np.ndarray:
        """Make predictions using the trained model"""
        try:
            # Load model
            model_path = f'models/real_money_{symbol.lower()}_model.pkl'
            if not Path(model_path).exists():
                backtest_logger.error(f"No trained model found for {symbol}")
                return np.zeros(len(df))
            
            model = joblib.load(model_path)
            
            # Prepare features
            feature_columns = [
                'rsi', 'macd', 'macd_signal', 'macd_diff',
                'sma_20', 'volume_ratio', 'atr'
            ]
            
            X = df[feature_columns].values
            
            # Make predictions
            return model.predict(X)
            
        except Exception as e:
            backtest_logger.error(f"Error making predictions: {e}")
            return np.zeros(len(df))
    
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
            # Create DataFrame from results
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
            exit_trades = trades_df[trades_df['type'] == 'exit']
            plt.hist(exit_trades['pnl'], bins=30, edgecolor='black')
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
            win_trades = exit_trades[exit_trades['pnl'] > 0]
            loss_trades = exit_trades[exit_trades['pnl'] <= 0]
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
            backtest_logger.error(f"Error generating plots: {e}")
            backtest_logger.warning("Continuing without plots...")
    
    def _save_results(self):
        """Save backtest results to file"""
        try:
            # Save detailed results
            results_df = pd.DataFrame(self.results['trades'])
            results_df.to_csv('results/backtest_trades.csv', index=False)
            
            # Save daily PnL
            pnl_df = pd.DataFrame(self.results['daily_pnl'])
            pnl_df.to_csv('results/daily_pnl.csv', index=False)
            
            # Calculate additional metrics
            exit_trades = results_df[results_df['type'] == 'exit']
            win_trades = exit_trades[exit_trades['pnl'] > 0]
            loss_trades = exit_trades[exit_trades['pnl'] <= 0]
            
            avg_win = win_trades['pnl'].mean() if len(win_trades) > 0 else 0
            avg_loss = loss_trades['pnl'].mean() if len(loss_trades) > 0 else 0
            profit_factor = abs(win_trades['pnl'].sum() / loss_trades['pnl'].sum()) if len(loss_trades) > 0 else float('inf')
            
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
            backtest_logger.error(f"Error saving results: {e}")

def main():
    # Example usage
    symbols = [
        'AMBUSDT',  # Best performer
        'ARUSDT',   # Second best
    ]
    
    backtester = Backtester(
        symbols=symbols,
        initial_capital=3.0,     # Start with $3 USDT
        position_size_pct=0.3,   # Use 30% of capital per trade
        stop_loss_pct=0.015,     # 1.5% stop loss
        take_profit_pct=0.045,   # 4.5% take profit
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