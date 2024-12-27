import logging
import os
from logging.handlers import RotatingFileHandler
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Email configuration
EMAIL_RECIPIENT = os.getenv('EMAIL_RECIPIENT')

class EmailHandler(logging.Handler):
    """Custom logging handler that sends emails for important events"""
    
    def __init__(self, notification_types=None):
        super().__init__()
        self.notification_types = notification_types or ['TRADE', 'CRITICAL', 'ERROR']
    
    def emit(self, record):
        try:
            # Check if this is a notification we want to send
            if not any(ntype in record.msg for ntype in self.notification_types):
                return
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = "Binance Trading Bot"
            msg['To'] = EMAIL_RECIPIENT
            
            # Set subject based on log type
            if 'TRADE' in record.msg:
                msg['Subject'] = "Trade Execution Notification"
            elif 'CRITICAL' in record.msg or 'ERROR' in record.msg:
                msg['Subject'] = "Trading Bot Alert - Critical Event"
            else:
                msg['Subject'] = "Trading Bot Notification"
            
            # Format the email body
            body = f"""
            Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            Level: {record.levelname}
            
            Message:
            {record.msg}
            
            Additional Info:
            {getattr(record, 'trade_info', '')}
            {getattr(record, 'error_info', '')}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email using local SMTP server
            with smtplib.SMTP('localhost') as server:
                server.send_message(msg)
            
        except Exception as e:
            print(f"Error sending email notification: {e}")

def setup_logger(name, log_file, level=logging.INFO):
    """Set up logger with file and console handlers"""
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # File handler (rotating log files)
    file_handler = RotatingFileHandler(
        f'logs/{log_file}',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level)
    
    # Email handler
    email_handler = EmailHandler()
    email_handler.setLevel(logging.WARNING)  # Only send emails for warning and above
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.addHandler(email_handler)
    
    return logger

def get_logger(name):
    """Get or create a logger for a specific component"""
    if name == 'trading':
        return setup_logger('trading', 'trading.log')
    elif name == 'prediction':
        return setup_logger('prediction', 'prediction.log')
    elif name == 'binance':
        return setup_logger('binance', 'binance.log')
    elif name == 'bot':
        return setup_logger('bot', 'bot.log')
    else:
        return setup_logger(name, f'{name}.log')

# Extra logging functions for trade notifications
def log_trade_execution(logger, trade_info: dict):
    """Log trade execution with email notification"""
    message = "[TRADE] New Trade Executed\n"
    message += f"Symbol: {trade_info['symbol']}\n"
    message += f"Side: {trade_info['side']}\n"
    message += f"Entry Price: ${trade_info['entry_price']:.8f}\n"
    message += f"Position Size: {trade_info['position_size']:.8f}\n"
    message += f"Stop Loss: ${trade_info['stop_loss']:.8f}\n"
    message += f"Take Profit: ${trade_info['take_profit']:.8f}\n"
    
    logger.warning(message, extra={
        'trade_info': trade_info
    })

def log_trade_exit(logger, exit_info: dict):
    """Log trade exit with email notification"""
    message = "[TRADE] Position Closed\n"
    message += f"Symbol: {exit_info['symbol']}\n"
    message += f"Side: {exit_info['side']}\n"
    message += f"Entry Price: ${exit_info['entry_price']:.8f}\n"
    message += f"Exit Price: ${exit_info['exit_price']:.8f}\n"
    message += f"PnL: ${exit_info['pnl']:.2f}\n"
    message += f"Holding Time: {exit_info['holding_time']}\n"
    message += f"Exit Reason: {exit_info['exit_reason']}\n"
    
    logger.warning(message, extra={
        'trade_info': exit_info
    })

def log_critical_error(logger, error_info: dict):
    """Log critical error with email notification"""
    message = "[CRITICAL] Trading Bot Error\n"
    message += f"Error Type: {error_info['error_type']}\n"
    message += f"Error Message: {error_info['error_message']}\n"
    message += f"Component: {error_info['component']}\n"
    if 'symbol' in error_info:
        message += f"Symbol: {error_info['symbol']}\n"
    if 'additional_info' in error_info:
        message += f"Additional Info: {error_info['additional_info']}\n"
    
    logger.critical(message, extra={
        'error_info': error_info
    }) 