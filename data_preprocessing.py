import json
import pandas as pd
import numpy as np
from datetime import datetime
import ta

def load_historical_data():
    """Load data from historical_data.json"""
    try:
        with open('historical_data.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: historical_data.json not found")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in historical_data.json")
        return None

def convert_to_dataframe(symbol_data):
    """Convert JSON data to pandas DataFrame"""
    df = pd.DataFrame(symbol_data['data'])
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['time'], unit='ms')
    
    # Set timestamp as index
    df.set_index('timestamp', inplace=True)
    
    # Drop original time column
    df.drop('time', axis=1, inplace=True)
    
    return df

def add_technical_indicators(df):
    """Add technical analysis indicators to DataFrame"""
    
    # Initialize indicator objects
    rsi = ta.momentum.RSIIndicator(df['close'], window=14)
    macd = ta.trend.MACD(df['close'])
    bb = ta.volatility.BollingerBands(df['close'])
    
    # Add RSI
    df['rsi'] = rsi.rsi()
    
    # Add MACD
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    # Add Bollinger Bands
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_mid'] = bb.bollinger_mavg()
    
    # Add Simple Moving Averages
    df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
    df['sma_200'] = ta.trend.sma_indicator(df['close'], window=200)
    
    # Add Average True Range (ATR)
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    
    # Add price momentum
    df['price_momentum'] = df['close'].pct_change()
    
    # Add volume indicators
    df['volume_sma'] = ta.trend.sma_indicator(df['volume'], window=20)
    df['volume_momentum'] = df['volume'].pct_change()
    
    return df

def clean_data(df):
    """Clean the DataFrame by handling missing values and outliers"""
    
    # Forward fill missing values using newer method
    df = df.ffill()
    
    # Backward fill any remaining missing values using newer method
    df = df.bfill()
    
    # Remove rows with infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    return df

def process_all_data():
    """Process all symbol data and save to processed_data.json"""
    # Load historical data
    historical_data = load_historical_data()
    if not historical_data:
        return
    
    processed_data = {}
    
    for symbol, data in historical_data.items():
        print(f"Processing {symbol}...")
        
        # Convert to DataFrame
        df = convert_to_dataframe(data)
        
        # Add technical indicators
        df = add_technical_indicators(df)
        
        # Clean data
        df = clean_data(df)
        
        # Convert DataFrame back to dictionary with proper datetime handling
        df_dict = df.reset_index().to_dict(orient='records')
        
        # Convert datetime objects to ISO format strings
        for record in df_dict:
            record['timestamp'] = record['timestamp'].isoformat()
        
        processed_data[symbol] = {
            'interval': data['interval'],
            'last_updated': datetime.now().isoformat(),
            'data': df_dict
        }
    
    # Save processed data to JSON
    try:
        with open('processed_data.json', 'w') as f:
            json.dump(processed_data, f, indent=2)
        print("\nProcessed data successfully saved to processed_data.json")
    except Exception as e:
        print(f"Error saving processed data: {e}")

def main():
    print("Starting data preprocessing...")
    process_all_data()

if __name__ == "__main__":
    main() 