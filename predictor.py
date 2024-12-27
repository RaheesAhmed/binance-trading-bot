import joblib
import pandas as pd
import numpy as np
import logging
from binance_connection import binance_connection
from logger_setup import get_logger
import json
import os
from typing import Optional, Dict, Callable
from websocket import WebSocketApp
import threading
from datetime import datetime
import queue
from collections import deque
import time

# Get logger
prediction_logger = get_logger('prediction')

class PricePredictor:
    def __init__(self):
        """Initialize the predictor with trained model and scaler"""
        try:
            # Load model and related files
            self.model = joblib.load('models/trading_model.pkl')
            self.scaler = joblib.load('models/feature_scaler.pkl')
            with open('data/feature_names.json', 'r') as f:
                self.feature_names = json.load(f)['features']
            
            # Initialize WebSocket components
            self.ws = None
            self.ws_thread = None
            self.running = False
            self.data_queue = queue.Queue()
            
            # Store recent candles for each symbol
            self.candle_history = {}  # symbol -> deque of candles
            self.callbacks = {}       # symbol -> list of callback functions
            
            prediction_logger.info("Successfully loaded model, scaler, and feature names")
        except Exception as e:
            prediction_logger.error(f"Error initializing predictor: {e}")
            raise
    
    def start_symbol_stream(self, symbol: str, callback: Optional[Callable] = None):
        """Start streaming data for a symbol"""
        try:
            # Initialize candle history with historical data first
            klines = binance_connection.get_historical_klines(symbol, '1h', 200)
            if not klines:
                prediction_logger.error(f"Could not initialize history for {symbol}")
                return False
            
            self.candle_history[symbol] = deque(maxlen=200)  # Keep last 200 candles
            for k in klines:
                self.candle_history[symbol].append({
                    'timestamp': pd.to_datetime(k['time'], unit='ms'),
                    'open': float(k['open']),
                    'high': float(k['high']),
                    'low': float(k['low']),
                    'close': float(k['close']),
                    'volume': float(k['volume'])
                })
            
            # Store callback if provided
            if callback:
                if symbol not in self.callbacks:
                    self.callbacks[symbol] = []
                self.callbacks[symbol].append(callback)
            
            # Start WebSocket if not already running
            if not self.running:
                self._start_websocket()
            
            prediction_logger.info(f"Started streaming for {symbol}")
            return True
            
        except Exception as e:
            prediction_logger.error(f"Error starting stream for {symbol}: {e}")
            return False
    
    def _start_websocket(self):
        """Start WebSocket connection"""
        try:
            base_url = "wss://testnet.binance.vision/ws" if binance_connection.use_testnet else "wss://stream.binance.com:9443/ws"
            
            # Create subscription message for all symbols
            streams = []
            for symbol in self.candle_history.keys():
                streams.append(f"{symbol.lower()}@kline_1h")
            
            def on_message(ws, message):
                try:
                    data = json.loads(message)
                    
                    # Handle subscription response
                    if 'result' in data:
                        prediction_logger.info(f"Subscription response: {data}")
                        return
                    
                    # Validate message structure
                    if not all(key in data for key in ['e', 's']):
                        prediction_logger.warning(f"Received incomplete message: {data}")
                        return
                    
                    # Handle kline messages
                    if data['e'] == 'kline':
                        if 'k' not in data:
                            prediction_logger.warning(f"Kline data missing in message: {data}")
                            return
                            
                        symbol = data['s']
                        k = data['k']
                        
                        # Validate kline data
                        required_fields = ['t', 'o', 'h', 'l', 'c', 'v', 'x']
                        if not all(field in k for field in required_fields):
                            prediction_logger.warning(f"Incomplete kline data for {symbol}: {k}")
                            return
                        
                        # Only process completed candles
                        if k['x']:  # Candle closed
                            candle = {
                                'timestamp': pd.to_datetime(k['t'], unit='ms'),
                                'open': float(k['o']),
                                'high': float(k['h']),
                                'low': float(k['l']),
                                'close': float(k['c']),
                                'volume': float(k['v'])
                            }
                            
                            # Update candle history
                            if symbol in self.candle_history:
                                self.candle_history[symbol].append(candle)
                                
                                # Make new prediction
                                prediction = self._predict_realtime(symbol)
                                
                                # Call callbacks
                                if symbol in self.callbacks:
                                    for callback in self.callbacks[symbol]:
                                        callback(prediction)
                            else:
                                prediction_logger.warning(f"Received data for untracked symbol: {symbol}")
                    else:
                        prediction_logger.debug(f"Ignored non-kline message: {data['e']}")
                        
                except json.JSONDecodeError as e:
                    prediction_logger.error(f"Failed to decode WebSocket message: {e}")
                except Exception as e:
                    prediction_logger.error(f"Error processing WebSocket message: {str(e)}", exc_info=True)
            
            def on_error(ws, error):
                prediction_logger.error(f"WebSocket error: {error}")
            
            def on_close(ws, close_status_code, close_msg):
                prediction_logger.warning("WebSocket connection closed")
                self.running = False
                
                # Attempt to reconnect after a delay
                threading.Timer(5.0, self._start_websocket).start()
            
            def on_open(ws):
                prediction_logger.info("WebSocket connection opened")
                self.running = True
                
                # Subscribe to all symbol streams
                subscribe_message = {
                    "method": "SUBSCRIBE",
                    "params": streams,
                    "id": 1
                }
                ws.send(json.dumps(subscribe_message))
            
            # Initialize WebSocket
            self.ws = WebSocketApp(
                base_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
            )
            
            # Start WebSocket in a separate thread
            self.ws_thread = threading.Thread(
                target=self.ws.run_forever,
                kwargs={"ping_interval": 60, "ping_timeout": 30}
            )
            self.ws_thread.daemon = True
            self.ws_thread.start()
            
        except Exception as e:
            prediction_logger.error(f"Error starting WebSocket: {e}")
    
    def _predict_realtime(self, symbol: str) -> Dict:
        """Make prediction using real-time data"""
        try:
            # Convert candle history to DataFrame
            df = pd.DataFrame(list(self.candle_history[symbol]))
            df.set_index('timestamp', inplace=True)
            
            # Calculate technical indicators
            df = self.calculate_technical_indicators(df)
            
            # Handle any NaN values
            df = df.ffill().bfill()
            
            # Prepare features
            features = self.prepare_features(df)
            if features is None:
                return {'success': False, 'error': 'Failed to prepare features'}
            
            # Make prediction
            predictions, confidences = self.predict(features)
            prediction = predictions[-1]
            confidence = confidences[-1]
            
            current_price = float(df['close'].iloc[-1])
            
            result = {
                'success': True,
                'symbol': symbol,
                'prediction': int(prediction),
                'confidence': float(confidence),
                'current_price': current_price,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            prediction_logger.info(f"Real-time prediction for {symbol}: {result}")
            return result
            
        except Exception as e:
            prediction_logger.error(f"Error making real-time prediction for {symbol}: {e}")
            return {'success': False, 'error': str(e)}
    
    def stop_symbol_stream(self, symbol: str):
        """Stop streaming for a symbol"""
        try:
            if symbol in self.candle_history:
                del self.candle_history[symbol]
            if symbol in self.callbacks:
                del self.callbacks[symbol]
            
            # If no more symbols, close WebSocket
            if not self.candle_history:
                self.stop_all_streams()
            
            prediction_logger.info(f"Stopped streaming for {symbol}")
            return True
            
        except Exception as e:
            prediction_logger.error(f"Error stopping stream for {symbol}: {e}")
            return False
    
    def stop_all_streams(self):
        """Stop all WebSocket streams"""
        try:
            if self.ws:
                self.ws.close()
            self.running = False
            self.candle_history.clear()
            self.callbacks.clear()
            prediction_logger.info("Stopped all streams")
            return True
            
        except Exception as e:
            prediction_logger.error(f"Error stopping streams: {e}")
            return False
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for prediction"""
        try:
            # RSI (standard and slow)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0))
            loss = (-delta.where(delta < 0, 0))
            
            # Standard RSI (14 periods)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Slow RSI (21 periods)
            avg_gain_slow = gain.rolling(window=21).mean()
            avg_loss_slow = loss.rolling(window=21).mean()
            rs_slow = avg_gain_slow / avg_loss_slow
            df['rsi_slow'] = 100 - (100 / (1 + rs_slow))
            
            # MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_diff'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            df['bb_mid'] = df['close'].rolling(window=bb_period).mean()
            bb_std_dev = df['close'].rolling(window=bb_period).std()
            df['bb_high'] = df['bb_mid'] + (bb_std * bb_std_dev)
            df['bb_low'] = df['bb_mid'] - (bb_std * bb_std_dev)
            df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
            
            # Moving Averages
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['sma_200'] = df['close'].rolling(window=200).mean()
            df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
            
            # ATR (Average True Range)
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            df['atr'] = ranges.max(axis=1).rolling(14).mean()
            
            # Enhanced features
            df['price_change'] = df['close'].pct_change()
            df['volume_change'] = df['volume'].pct_change()
            
            # Price momentum features
            for period in [5, 10, 20]:
                df[f'momentum_{period}'] = df['close'].pct_change(periods=period)
                df[f'volume_momentum_{period}'] = df['volume'].pct_change(periods=period)
            
            # Volatility features
            df['volatility'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean()
            df['high_low_range'] = (df['high'] - df['low']) / df['close']
            
            # Trend features
            df['trend_strength'] = abs(df['ema_9'] - df['sma_20']) / df['close']
            df['trend_direction'] = np.where(df['ema_9'] > df['sma_20'], 1, -1)
            
            # Price position and range
            df['price_range'] = (df['high'] - df['low']) / df['close']
            df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            
            return df
            
        except Exception as e:
            prediction_logger.error(f"Error calculating technical indicators: {e}")
            raise
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction"""
        try:
            # Calculate technical indicators
            features = pd.DataFrame(index=df.index)
            
            # Ensure all required features are present and in correct order
            for feature in self.feature_names:
                if feature not in df.columns:
                    prediction_logger.warning(f"Missing feature: {feature}")
                    return None
                features[feature] = df[feature]
            
            # Scale features
            scaled_features = self.scaler.transform(features)
            return pd.DataFrame(scaled_features, columns=self.feature_names, index=features.index)
            
        except Exception as e:
            prediction_logger.error(f"Error preparing features: {e}")
            return None
    
    def predict(self, df: pd.DataFrame) -> tuple:
        """Make predictions using the model"""
        try:
            # Make predictions
            predictions = self.model.predict(df)
            probabilities = self.model.predict_proba(df)
            confidences = np.max(probabilities, axis=1)
            
            prediction_logger.info(f"Prediction distribution: {pd.Series(predictions).value_counts().to_dict()}")
            
            return predictions, confidences
            
        except Exception as e:
            prediction_logger.error(f"Error making prediction: {e}")
            return [], []
    
    def make_prediction(self, symbol: str) -> Dict:
        """Make prediction for a given symbol
        Returns:
            Dict containing prediction (0 or 1) and confidence score
        """
        try:
            # Get historical klines for technical indicator calculation
            klines = binance_connection.get_historical_klines(symbol, '1h', 200)
            if not klines:
                logging.warning(f"No data available for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame([{
                'timestamp': pd.to_datetime(k['time'], unit='ms'),
                'open': float(k['open']),
                'high': float(k['high']),
                'low': float(k['low']),
                'close': float(k['close']),
                'volume': float(k['volume'])
            } for k in klines])
            
            df.set_index('timestamp', inplace=True)
            
            # Calculate technical indicators
            df = self.calculate_technical_indicators(df)
            
            # Handle any NaN values
            df = df.ffill().bfill()
            
            # Prepare features
            features = self.prepare_features(df)
            if features is None:
                return {'success': False, 'error': 'Failed to prepare features'}
            
            # Make prediction
            predictions, confidences = self.predict(features)
            prediction = predictions[-1]
            confidence = confidences[-1]
            
            current_price = binance_connection.get_symbol_price(symbol)
            
            result = {
                'success': True,
                'symbol': symbol,
                'prediction': int(prediction),  # 0 for price drop, 1 for price increase
                'confidence': float(confidence),
                'current_price': current_price,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            logging.info(f"Prediction for {symbol}: {result}")
            return result
            
        except Exception as e:
            logging.error(f"Error making prediction for {symbol}: {e}")
            return {'success': False, 'error': str(e)}

# Create singleton instance
predictor = PricePredictor()

def start_streaming(symbol: str, callback: Optional[Callable] = None) -> bool:
    """Start streaming predictions for a symbol"""
    return predictor.start_symbol_stream(symbol, callback)

def stop_streaming(symbol: str) -> bool:
    """Stop streaming predictions for a symbol"""
    return predictor.stop_symbol_stream(symbol)

def make_prediction(symbol: str) -> Dict:
    """Make a one-time prediction for a symbol"""
    return predictor._predict_realtime(symbol) if symbol in predictor.candle_history else predictor.make_prediction(symbol)

# Example usage
if __name__ == "__main__":
    def print_prediction(prediction: Dict):
        if prediction['success']:
            print(f"\nNew prediction for {prediction['symbol']}:")
            print(f"Direction: {'UP' if prediction['prediction'] == 1 else 'DOWN'}")
            print(f"Confidence: {prediction['confidence']:.2%}")
            print(f"Price: {prediction['current_price']}")
        else:
            print(f"\nError: {prediction['error']}")

    # Start streaming for test symbols
    test_symbols = [
        'AMBUSDT',  # Best performer from backtest
        'ARUSDT',   # Second best from backtest
        'AIUSDT'    # Third best from backtest
    ]
    
    for symbol in test_symbols:
        start_streaming(symbol, print_prediction)
    
    # Keep script running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping streams...")
        for symbol in test_symbols:
            stop_streaming(symbol)