from binance_connection import binance_connection
from datetime import datetime

def format_price(price):
    return f"${price:,.2f}" if price else "N/A"

def main():
    # Get account information
    account_info = binance_connection.get_account_info()
    
    print("\n=== Binance Account Information ===")
    print(f"Environment: {'Testnet' if binance_connection.use_testnet else 'Production'}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nBalances:")
    print(f"Total Balance: {account_info['totalBalance']:,.2f} USDT")
    print(f"Available Balance: {account_info['availableBalance']:,.2f} USDT")
    print(f"Locked Balance: {account_info['totalBalance'] - account_info['availableBalance']:,.2f} USDT")
    print(f"Open Orders: {account_info['openOrders']}")
    
    # Get current prices for major pairs
    pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    print("\nCurrent Prices:")
    for pair in pairs:
        price = binance_connection.get_symbol_price(pair)
        print(f"{pair}: {format_price(price)}")

    # Get some historical data for BTC
    print("\nRecent BTC/USDT Candlesticks (last 5):")
    klines = binance_connection.get_historical_klines('BTCUSDT', '1h', 5)
    for kline in klines:
        dt = datetime.fromtimestamp(kline['time']/1000).strftime('%Y-%m-%d %H:%M')
        print(f"Time: {dt}")
        print(f"Open: {format_price(kline['open'])}")
        print(f"High: {format_price(kline['high'])}")
        print(f"Low: {format_price(kline['low'])}")
        print(f"Close: {format_price(kline['close'])}")
        print(f"Volume: {kline['volume']:,.8f} BTC")
        print("---")

if __name__ == "__main__":
    main() 