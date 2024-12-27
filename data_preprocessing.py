import json
import pandas as pd
import numpy as np
from datetime import datetime
import ta
from ta.trend import MACD, SMAIndicator, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, AccDistIndexIndicator
from ta.others import DailyReturnIndicator

def load_historical_data():
    """Load data from historical_data.json"""
    try:
        with open('data/historical_data.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: historical_data.json not found")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in historical_data.json")
        return None

def convert_to_dataframe(symbol_data, interval):
    """Convert JSON data to pandas DataFrame for a specific interval"""
    if interval not in symbol_data:
        return None
        
    df = pd.DataFrame(symbol_data[interval]['data'])
    
    # First check how many columns we have
    num_columns = len(df.columns)
    
    if num_columns == 12:
        # Full kline data format
        df.columns = ['time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
                     'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore']
    elif num_columns == 6:
        # Simplified format
        df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
    else:
        print(f"Unexpected number of columns: {num_columns}")
        return None
    
    # Convert types for the columns we know we need
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    if 'quote_volume' in df.columns:
        numeric_columns.extend(['quote_volume', 'taker_buy_base', 'taker_buy_quote'])
    
    df[numeric_columns] = df[numeric_columns].astype(float)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['time'].astype(float), unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Drop unnecessary columns
    columns_to_drop = ['time']
    if 'close_time' in df.columns:
        columns_to_drop.extend(['close_time', 'ignore'])
    df.drop(columns_to_drop, axis=1, inplace=True)
    
    return df

def add_technical_indicators(df):
    """Add comprehensive technical analysis indicators to DataFrame"""
    
    # Price action features
    df['price_range'] = (df['high'] - df['low']) / df['close']
    df['body_size'] = abs(df['close'] - df['open']) / df['close']
    df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
    df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
    
    # Trend Indicators
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    # Multiple timeframe SMAs
    for period in [5, 8, 13, 21, 34, 55, 89]:
        df[f'sma_{period}'] = SMAIndicator(df['close'], window=period).sma_indicator()
    
    # Ichimoku Cloud
    ichimoku = IchimokuIndicator(high=df['high'], low=df['low'])
    df['ichimoku_a'] = ichimoku.ichimoku_a()
    df['ichimoku_b'] = ichimoku.ichimoku_b()
    df['ichimoku_base'] = ichimoku.ichimoku_base_line()
    df['ichimoku_conv'] = ichimoku.ichimoku_conversion_line()
    
    # Momentum Indicators
    rsi = RSIIndicator(df['close'])
    df['rsi'] = rsi.rsi()
    
    stoch = StochasticOscillator(df['high'], df['low'], df['close'])
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    # Volatility Indicators
    bb = BollingerBands(df['close'])
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_mid'] = bb.bollinger_mavg()
    df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
    
    atr = AverageTrueRange(df['high'], df['low'], df['close'])
    df['atr'] = atr.average_true_range()
    df['atr_pct'] = df['atr'] / df['close']
    
    # Volume Indicators
    obv = OnBalanceVolumeIndicator(df['close'], df['volume'])
    df['obv'] = obv.on_balance_volume()
    
    adi = AccDistIndexIndicator(df['high'], df['low'], df['close'], df['volume'])
    df['adi'] = adi.acc_dist_index()
    
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_std'] = df['volume'].rolling(window=20).std()
    df['volume_zscore'] = (df['volume'] - df['volume_sma']) / df['volume_std']
    
    # Returns and Momentum
    returns = DailyReturnIndicator(df['close'])
    df['returns'] = returns.daily_return()
    
    # Momentum features with different periods
    for period in [5, 10, 15, 20]:
        df[f'momentum_{period}'] = df['close'].pct_change(periods=period)
        df[f'volume_momentum_{period}'] = df['volume'].pct_change(periods=period)
    
    return df

def clean_data(df):
    """Clean the DataFrame by handling missing values and outliers"""
    
    # Forward fill missing values
    df = df.ffill()
    
    # Backward fill any remaining missing values
    df = df.bfill()
    
    # Handle outliers using IQR method for volume and returns
    for col in ['volume', 'returns']:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    # Remove rows with infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    return df

def process_all_data():
    """Process all symbol data and save to processed_data.json"""
    historical_data = load_historical_data()
    if not historical_data:
        return
    
    processed_data = {}
    
    for symbol, timeframes_data in historical_data.items():
        print(f"Processing {symbol}...")
        symbol_processed = {}
        
        for interval in timeframes_data.keys():
            # Convert to DataFrame
            df = convert_to_dataframe(timeframes_data, interval)
            if df is None:
                continue
                
            # Add technical indicators
            df = add_technical_indicators(df)
            
            # Clean data
            df = clean_data(df)
            
            # Convert DataFrame back to dictionary
            df_dict = df.reset_index().to_dict(orient='records')
            
            # Convert datetime objects to ISO format strings
            for record in df_dict:
                record['timestamp'] = record['timestamp'].isoformat()
            
            symbol_processed[interval] = {
                'interval': interval,
                'last_updated': datetime.now().isoformat(),
                'data': df_dict
            }
        
        processed_data[symbol] = symbol_processed
    
    # Save processed data
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'data/processed_data_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(processed_data, f, indent=2)
            
        # Update main processed data file
        with open('data/processed_data.json', 'w') as f:
            json.dump(processed_data, f, indent=2)
            
        print(f"\nProcessed data successfully saved to {filename} and processed_data.json")
    except Exception as e:
        print(f"Error saving processed data: {e}")

def main():
    print("Starting data preprocessing...")
    process_all_data()

if __name__ == "__main__":
    main() 