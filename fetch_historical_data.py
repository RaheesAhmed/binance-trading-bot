from binance_connection import binance_connection
import json
from datetime import datetime
import time

def get_volatile_coins(min_volume_btc=1000):
    """Get list of volatile coins based on 24h volume"""
    try:
        # Get 24h ticker for all symbols
        tickers = binance_connection.client.get_ticker()
        
        # Filter for USDT pairs with significant volume
        volatile_pairs = [
            ticker['symbol'] for ticker in tickers
            if ticker['symbol'].endswith('USDT')
            and float(ticker['volume']) * float(ticker['lastPrice']) >= min_volume_btc
            and float(ticker['priceChangePercent']) >= 5  # 5% price change in 24h
        ]
        
        return sorted(volatile_pairs)[:10]  # Return top 10 volatile pairs
    except Exception as e:
        print(f"Error fetching volatile coins: {e}")
        return []

def fetch_and_store_historical_data(symbols, interval='1h', limit=1000):
    """Fetch historical data for given symbols and store in JSON"""
    historical_data = {}
    
    for symbol in symbols:
        print(f"Fetching data for {symbol}...")
        try:
            # Get historical klines
            klines = binance_connection.get_historical_klines(symbol, interval, limit)
            
            # Format data for storage
            historical_data[symbol] = {
                'interval': interval,
                'last_updated': datetime.now().isoformat(),
                'data': klines
            }
            
            # Add delay to avoid hitting rate limits
            time.sleep(1)
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            continue
    
    # Store data in JSON file
    try:
        with open('historical_data.json', 'w') as f:
            json.dump(historical_data, f, indent=2)
        print(f"\nData successfully stored in historical_data.json")
    except Exception as e:
        print(f"Error saving data to file: {e}")

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