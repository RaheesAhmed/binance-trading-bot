import json
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import os

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('graphs', exist_ok=True)
os.makedirs('logs', exist_ok=True)

class TradingBacktest:
    def __init__(self, 
                 initial_capital: float = 10000.0,
                 position_size: float = 0.1,  # 10% of capital per trade
                 trading_fee: float = 0.001):  # 0.1% trading fee
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.trading_fee = trading_fee
        
        # Load model and related files
        self.model = joblib.load('models/trading_model.pkl')
        self.scaler = joblib.load('models/feature_scaler.pkl')
        with open('data/feature_names.json', 'r') as f:
            self.feature_names = json.load(f)['features']
            
    def load_data(self) -> Dict:
        """Load processed data from JSON"""
        with open('data/processed_data.json', 'r') as f:
            return json.load(f)
            
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for prediction"""
        X = df[self.feature_names].copy()
        return self.scaler.transform(X)
        
    def calculate_metrics(self, returns: pd.Series) -> Dict:
        """Calculate trading performance metrics"""
        total_return = (returns + 1).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        
        # Calculate Sharpe Ratio (assuming risk-free rate of 2%)
        risk_free_rate = 0.02
        excess_returns = returns - risk_free_rate/252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        
        # Calculate Maximum Drawdown
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        # Calculate Win Rate
        winning_trades = (returns > 0).sum()
        total_trades = len(returns[returns != 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades
        }
        
    def plot_equity_curve(self, returns: pd.Series, symbol: str):
        """Plot equity curve for a symbol"""
        cumulative_returns = (1 + returns).cumprod()
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative_returns.index, cumulative_returns.values)
        plt.title(f'Equity Curve - {symbol}')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value (Starting = 1)')
        plt.grid(True)
        plt.savefig(f'graphs/equity_curve_{symbol}.png')
        plt.close('all')  # Close all figures to free memory
        
    def backtest_symbol(self, df: pd.DataFrame, symbol: str) -> Tuple[pd.Series, Dict]:
        """Backtest trading strategy for a single symbol"""
        # Prepare features and make predictions
        X = self.prepare_features(df)
        predictions = self.model.predict(X)
        
        # Initialize position and returns series
        position = 0  # -1 for short, 0 for neutral, 1 for long
        returns = pd.Series(index=df.index[:-1], dtype=float)
        
        # Calculate returns for each period
        capital = self.initial_capital
        
        for i in range(len(predictions)):
            current_price = df.iloc[i]['close']
            
            # Determine position based on prediction
            new_position = 1 if predictions[i] == 1 else -1
            
            # Calculate transaction costs if position changes
            if new_position != position:
                # Close existing position if any
                if position != 0:
                    transaction_cost = abs(position) * capital * self.position_size * self.trading_fee
                    capital -= transaction_cost
                
                # Open new position
                transaction_cost = abs(new_position) * capital * self.position_size * self.trading_fee
                capital -= transaction_cost
                
            # Update position
            position = new_position
            
            # Calculate returns
            if i < len(predictions) - 1:
                next_price = df.iloc[i + 1]['close']
                price_return = (next_price - current_price) / current_price
                position_return = position * price_return * self.position_size
                returns.iloc[i] = position_return
        
        # Calculate metrics
        metrics = self.calculate_metrics(returns)
        
        return returns, metrics
        
    def run_backtest(self):
        """Run backtest across all symbols"""
        processed_data = self.load_data()
        all_metrics = {}
        returns_list = []  # List to collect all returns series
        
        # Create log file for detailed results
        log_file = f'logs/backtest_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        with open(log_file, 'w') as f:
            f.write("Backtest Results\n")
            f.write("=" * 50 + "\n\n")
            
            print("\nStarting backtest...")
            f.write("Starting backtest...\n")
            
            for symbol, data in processed_data.items():
                print(f"\nBacktesting {symbol}...")
                f.write(f"\nBacktesting {symbol}...\n")
                
                # Convert data to DataFrame
                df = pd.DataFrame(data['data'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                
                # Run backtest for symbol
                returns, metrics = self.backtest_symbol(df, symbol)
                
                # Store results
                all_metrics[symbol] = metrics
                returns_list.append(returns)
                
                # Plot equity curve
                self.plot_equity_curve(returns, symbol)
                
                # Log results
                results_text = (
                    f"Results for {symbol}:\n"
                    f"Total Return: {metrics['total_return']:.2%}\n"
                    f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
                    f"Maximum Drawdown: {metrics['max_drawdown']:.2%}\n"
                    f"Win Rate: {metrics['win_rate']:.2%}\n"
                    f"Total Trades: {metrics['total_trades']}\n"
                )
                print(results_text)
                f.write(results_text + "\n")
            
            # Combine all returns series
            all_returns = pd.concat(returns_list) if returns_list else pd.Series(dtype=float)
            
            # Calculate portfolio metrics
            portfolio_metrics = self.calculate_metrics(all_returns)
            
            # Log portfolio results
            portfolio_text = (
                "\nOverall Portfolio Performance:\n"
                f"Total Return: {portfolio_metrics['total_return']:.2%}\n"
                f"Annual Return: {portfolio_metrics['annual_return']:.2%}\n"
                f"Sharpe Ratio: {portfolio_metrics['sharpe_ratio']:.2f}\n"
                f"Maximum Drawdown: {portfolio_metrics['max_drawdown']:.2%}\n"
                f"Win Rate: {portfolio_metrics['win_rate']:.2%}\n"
                f"Total Trades: {portfolio_metrics['total_trades']}\n"
            )
            print(portfolio_text)
            f.write(portfolio_text)
        
        # Save metrics to file
        results = {
            'portfolio_metrics': portfolio_metrics,
            'symbol_metrics': all_metrics,
            'backtest_date': datetime.now().isoformat()
        }
        
        with open('data/backtest_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Plot overall equity curve
        self.plot_equity_curve(all_returns, 'Portfolio')

def main():
    backtest = TradingBacktest()
    backtest.run_backtest()

if __name__ == "__main__":
    main() 