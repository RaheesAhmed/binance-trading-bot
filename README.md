# Binance Trading Bot

An advanced algorithmic trading bot for Binance cryptocurrency exchange, featuring machine learning-based predictions, real-time data streaming, and comprehensive risk management.

## Features

### Core Components

- **Machine Learning Predictions**: Uses trained models to predict price movements
- **Real-time Data Streaming**: WebSocket integration for live market data
- **Risk Management**: Advanced position sizing and risk control
- **Automated Trading**: Scheduled trade execution and position management
- **Email Notifications**: Real-time alerts for trades and critical events
- **Paper Trading**: Safe testing using Binance Testnet

### Trading Strategy

- Monitors selected new coins for significant price movements
- Makes predictions using technical indicators and machine learning
- Executes both long and short positions
- Implements stop-loss and take-profit mechanisms
- Time-based position management (24-hour maximum hold time)

### Risk Management Features

- Dynamic position sizing based on:
  - Account balance
  - Volatility
  - Stop loss distance
  - Risk per trade (1%)
- Maximum position size limits (5% of account)
- Daily loss limits (2% of capital)
- Maximum correlated positions (2 similar positions)
- Maximum simultaneous positions (3)
- Daily trade limits (10 trades)

### Technical Indicators

- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Moving Averages (20, 50, 200 periods)
- ATR (Average True Range)
- Price and Volume Momentum

### Monitoring and Notifications

- Real-time email notifications for:
  - Trade executions
  - Position closures
  - Critical errors
  - Daily summaries
- Comprehensive logging system
- Performance tracking and reporting

## Project Structure

```
binance-trading-bot/
├── bot.py                 # Main bot orchestration
├── predictor.py           # ML predictions and data streaming
├── trade_executor.py      # Trade execution and management
├── binance_connection.py  # Exchange connectivity
├── logger_setup.py        # Logging and notifications
├── models/               # Trained models and scalers
│   ├── trading_model.pkl
│   └── feature_scaler.pkl
├── data/                # Historical and processed data
│   ├── historical_data.json
│   └── feature_names.json
├── graphs/              # Performance visualizations
├── logs/               # Application logs
└── .env                # Configuration settings
```

## Configuration

### Environment Variables (.env)

```
USE_TESTNET=true
TESTNET_API_KEY=your_testnet_api_key
TESTNET_API_SECRET=your_testnet_api_secret
EMAIL_RECIPIENT=your_email@example.com
```

### Trading Parameters

- `initial_capital`: Starting capital (default: 10000.0 USDT)
- `position_size_pct`: Base position size (1% of capital)
- `min_confidence`: Minimum prediction confidence (60%)
- `stop_loss_pct`: Stop loss percentage (2%)
- `take_profit_pct`: Take profit percentage (6%)
- `max_positions`: Maximum simultaneous positions (3)
- `max_daily_trades`: Maximum trades per day (10)

### Risk Management Parameters

- `max_daily_loss_pct`: Maximum daily loss (2%)
- `max_position_size_pct`: Maximum position size (5%)
- `risk_per_trade_pct`: Risk per trade (1%)
- `max_correlated_positions`: Maximum similar positions (2)
- `volatility_lookback`: Days for volatility calculation (20)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/raheesahmed/binance-trading-bot.git
cd binance-trading-bot
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment variables:

```bash
cp .env.example .env
# Edit .env with your settings
```

4. Run the bot:

```bash
python bot.py
```

## Usage

### Starting the Bot

The bot can be started in paper trading mode (testnet):

```bash
python bot.py
```

### Monitoring

- Check logs in the `logs/` directory
- Monitor email notifications for trade executions and alerts
- View daily summaries for performance tracking

### Scheduled Jobs

- Trade execution: Every hour
- Position management: Every 5 minutes
- Daily reset: Midnight (00:00)

## Safety Features

1. **Testnet Enforcement**

   - Mandatory paper trading mode
   - Real money protection
   - Safe testing environment

2. **Risk Controls**

   - Daily loss limits
   - Position size limits
   - Maximum trade counts
   - Correlation controls

3. **Error Handling**
   - Comprehensive error catching
   - Automatic reconnection
   - Critical error notifications

## Performance Monitoring

1. **Trade Tracking**

   - Entry and exit prices
   - Position sizes
   - PnL calculations
   - Hold times

2. **Daily Statistics**

   - Total trades
   - Win/loss ratio
   - Daily PnL
   - Active positions

3. **Risk Metrics**
   - Volatility measurements
   - Risk exposure
   - Position correlation
   - Account balance tracking

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This bot is for educational purposes only. Cryptocurrency trading carries significant risks. Use at your own risk.
