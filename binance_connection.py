from binance.client import Client
from binance.exceptions import BinanceAPIException
import os
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/binance.log'),
        logging.StreamHandler()
    ]
)

# Create directories if they don't exist
os.makedirs('logs', exist_ok=True)

# Load environment variables
load_dotenv(override=True)  # Add override=True to ensure variables are loaded

class BinanceConnection:
    def __init__(self):
        # Debug log the environment variable
        use_testnet_value = os.getenv('USE_TESTNET')
        logging.info(f"USE_TESTNET value from env: {use_testnet_value}")
        
        # Ensure we're using testnet (more explicit check)
        self.use_testnet = str(use_testnet_value).lower() in ['true', '1', 'yes']
        logging.info(f"Testnet mode: {self.use_testnet}")
        
        if not self.use_testnet:
            logging.error("Production mode is not allowed! Please set USE_TESTNET=true in .env")
            raise ValueError("Must use testnet for paper trading")
        
        # Select API credentials based on environment
        api_key = os.getenv('TESTNET_API_KEY')
        api_secret = os.getenv('TESTNET_API_SECRET')
        
        # Debug log the API credentials (masked)
        logging.info(f"API Key found: {'Yes' if api_key else 'No'}")
        logging.info(f"API Secret found: {'Yes' if api_secret else 'No'}")
        
        if not api_key or not api_secret:
            raise ValueError("Missing TESTNET_API_KEY or TESTNET_API_SECRET in .env file")
        
        try:
            self.client = Client(api_key, api_secret, testnet=True)
            logging.info("Initialized Binance Testnet client successfully")
            
            # Test connection
            self.client.get_account()
            logging.info("Successfully connected to Binance Testnet")
        except BinanceAPIException as e:
            logging.error(f"Failed to connect to Binance Testnet: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error connecting to Binance Testnet: {e}")
            raise

    def get_account_info(self):
        """Get account information including balances"""
        try:
            # Get account information
            account = self.client.get_account()
            
            # Get USDT balance
            usdt_balance = next(
                (float(b['free']) for b in account['balances'] if b['asset'] == 'USDT'),
                0.0
            )
            
            # Get locked USDT
            usdt_locked = next(
                (float(b['locked']) for b in account['balances'] if b['asset'] == 'USDT'),
                0.0
            )
            
            # Get open orders
            open_orders = self.client.get_open_orders()
            
            return {
                'totalBalance': usdt_balance + usdt_locked,
                'availableBalance': usdt_balance,
                'openOrders': len(open_orders)
            }
        except BinanceAPIException as e:
            print(f"Error fetching account information: {e}")
            return {
                'totalBalance': 0.0,
                'availableBalance': 0.0,
                'openOrders': 0
            }

    def get_symbol_price(self, symbol):
        """Get current price for a symbol"""
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except BinanceAPIException as e:
            print(f"Error fetching price for {symbol}: {e}")
            return None

    def place_order(self, symbol, side, order_type, quantity=None, price=None, stopPrice=None):
        """Place an order on Binance"""
        try:
            params = {
                'symbol': symbol,
                'side': side,
                'type': order_type,
            }
            
            if quantity:
                params['quantity'] = quantity
            
            if price:
                params['price'] = price
                
            if stopPrice:
                params['stopPrice'] = stopPrice
            
            logging.info(f"Placing {order_type} order: {params}")
            
            if order_type == 'MARKET':
                if side == 'BUY':
                    order = self.client.order_market_buy(**params)
                else:
                    order = self.client.order_market_sell(**params)
            elif order_type == 'STOP_LOSS':
                if side == 'BUY':
                    order = self.client.create_order(**params)
                else:
                    order = self.client.create_order(**params)
            else:
                order = self.client.create_order(**params)
                
            logging.info(f"Order placed successfully: {order['orderId']}")
            return order
            
        except BinanceAPIException as e:
            logging.error(f"Error placing {order_type} order: {e}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error placing order: {e}")
            return None

    def get_historical_klines(self, symbol, interval, limit=100):
        """Get historical kline/candlestick data"""
        try:
            klines = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            return [{
                'time': k[0],
                'open': float(k[1]),
                'high': float(k[2]),
                'low': float(k[3]),
                'close': float(k[4]),
                'volume': float(k[5])
            } for k in klines]
        except BinanceAPIException as e:
            print(f"Error fetching klines for {symbol}: {e}")
            return []

# Create a singleton instance
binance_connection = BinanceConnection() 