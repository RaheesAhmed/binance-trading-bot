from binance_connection import binance_connection
import json
from datetime import datetime, timedelta
import time
import pandas as pd
import numpy as np

def get_volatile_coins(min_volume_usdt=1000000, min_price_change=3):
    """Get list of volatile coins based on multiple criteria"""
    try:
        # Get 24h ticker for all symbols
        tickers = binance_connection.client.get_ticker()
        
        # Convert to DataFrame for easier filtering
        df = pd.DataFrame(tickers)
        df['volume_usdt'] = df.apply(lambda x: float(x['volume']) * float(x['lastPrice']), axis=1)
        
        # Filter for USDT pairs
        df = df[df['symbol'].str.endswith('USDT')]
        
        # Apply multiple criteria
        volatile_pairs = df[
            (df['volume_usdt'] >= min_volume_usdt) &  # Minimum USDT volume
            (df['symbol'].str.contains('^(?!.*UP|.*DOWN|.*BULL|.*BEAR).*$')) &  # Exclude leveraged tokens
            (abs(df['priceChangePercent'].astype(float)) >= min_price_change)  # Minimum price change
        ]
        
        # Sort by volatility score (combination of volume and price change)
        volatile_pairs['volatility_score'] = (
            volatile_pairs['volume_usdt'].astype(float) * 
            abs(volatile_pairs['priceChangePercent'].astype(float))
        )
        volatile_pairs = volatile_pairs.sort_values('volatility_score', ascending=False)
        
        return volatile_pairs['symbol'].tolist()[:15]  # Return top 15 volatile pairs
        
    except Exception as e:
        print(f"Error fetching volatile coins: {e}")
        return []

def fetch_multi_timeframe_data(symbol, timeframes=['1m', '5m', '15m', '1h', '4h']):
    """Fetch data for multiple timeframes"""
    data = {}
    
    for interval in timeframes:
        try:
            # Calculate limit based on timeframe
            if interval.endswith('m'):
                limit = min(1000, int(24 * 60 / int(interval[:-1])))
            elif interval.endswith('h'):
                limit = min(1000, int(24 * 30 / int(interval[:-1])))
            else:
                limit = 1000
                
            klines = binance_connection.get_historical_klines(
                symbol, 
                interval, 
                limit=limit
            )
            
            data[interval] = {
                'interval': interval,
                'last_updated': datetime.now().isoformat(),
                'data': klines
            }
            
            time.sleep(0.5)  # Rate limiting
            
        except Exception as e:
            print(f"Error fetching {interval} data for {symbol}: {e}")
            continue
            
    return data

def fetch_and_store_historical_data(symbols):
    """Fetch historical data for given symbols and store in JSON"""
    historical_data = {}
    
    for symbol in symbols:
        print(f"Fetching multi-timeframe data for {symbol}...")
        try:
            historical_data[symbol] = fetch_multi_timeframe_data(symbol)
            print(f"Successfully fetched data for {symbol}")
            
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            continue
    
    # Store data in JSON file
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'data/historical_data_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(historical_data, f, indent=2)
        
        # Also update the main historical data file
        with open('data/historical_data.json', 'w') as f:
            json.dump(historical_data, f, indent=2)
            
        print(f"\nData successfully stored in {filename} and historical_data.json")
        
    except Exception as e:
        print(f"Error saving data to file: {e}")

def update_realtime_data(symbols, interval='1m'):
    """Update real-time data for given symbols"""
    try:
        # Load existing data
        with open('data/historical_data.json', 'r') as f:
            historical_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        historical_data = {}
    
    for symbol in symbols:
        try:
            # Get latest kline
            latest_kline = binance_connection.get_historical_klines(
                symbol, 
                interval, 
                limit=1
            )[0]
            
            # Update data
            if symbol in historical_data and interval in historical_data[symbol]:
                historical_data[symbol][interval]['data'].append(latest_kline)
                historical_data[symbol][interval]['last_updated'] = datetime.now().isoformat()
        
        except Exception as e:
            print(f"Error updating {symbol}: {e}")
            continue
    
    # Save updated data
    try:
        with open('data/historical_data.json', 'w') as f:
            json.dump(historical_data, f, indent=2)
    except Exception as e:
        print(f"Error saving updated data: {e}")

def main():
    print("Fetching volatile coins from Binance...")
    volatile_coins = get_volatile_coins()
    
    if not volatile_coins:
        print("No volatile coins found. Exiting...")
        return
    
    print(f"\nFound {len(volatile_coins)} volatile coins:")
    for coin in volatile_coins:
        print(f"- {coin}")
    
    print("\nFetching historical data...")
    fetch_and_store_historical_data(volatile_coins)

if __name__ == "__main__":
    main() 