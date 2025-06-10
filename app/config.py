"""
Configuration module for the trading algorithm.
Contains all parameters and environment variables.
"""
import os
from datetime import datetime, timedelta, date
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class for trading algorithm parameters."""
    
    def __init__(self):
        self.load_environment_variables()
        self.setup_trading_parameters()
        self.setup_backtesting_parameters()
        self.setup_data_parameters()
    
    def load_environment_variables(self):
        """Load API keys and credentials from environment variables."""
        # Alpaca credentials are read directly from environment variables
        # Set these environment variables:
        # - ALPACA_PAPER_KEY: Your paper trading API key
        # - ALPACA_PAPER_SECRET: Your paper trading secret key
        # - ALPACA_LIVE_KEY: Your live trading API key (optional)
        # - ALPACA_LIVE_SECRET: Your live trading secret key (optional)
        # - GOOGLE_APPLICATION_CREDENTIALS: Path to GCS service account JSON (optional)
        # - ENVIRONMENT: Environment setting (dev, qa, prod) - defaults to 'dev'
        
        # Environment setting (dev, qa, prod)
        self.ENVIRONMENT = os.getenv('ENVIRONMENT', 'dev').lower()
        if self.ENVIRONMENT not in ['dev', 'qa', 'prod']:
            print(f"Warning: Invalid ENVIRONMENT value '{self.ENVIRONMENT}'. Must be 'dev', 'qa', or 'prod'. Defaulting to 'dev'.")
            self.ENVIRONMENT = 'dev'
        
        required_vars = ['ALPACA_PAPER_KEY', 'ALPACA_PAPER_SECRET']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            print(f"Warning: Missing required environment variables: {missing_vars}")
            print("Set these environment variables for the algorithm to work properly.")
        
        # Optional variables
        optional_vars = ['ALPACA_LIVE_KEY', 'ALPACA_LIVE_SECRET', 'GOOGLE_APPLICATION_CREDENTIALS']
        for var in optional_vars:
            if not os.getenv(var):
                print(f"Info: Optional environment variable {var} not set.")
        
        print(f"Info: Running in '{self.ENVIRONMENT}' environment")
    
    def setup_trading_parameters(self):
        """Set up trading-related parameters."""
        # Trading mode
        self.PAPER_TRADE = os.getenv('PAPER_TRADE', 'True').lower() == 'true'
        
        # Portfolio management
        self.MAX_POSITIONS = int(os.getenv('MAX_POSITIONS', '10'))
        self.MAX_NEW_POSITIONS_PER_DAY = int(os.getenv('MAX_NEW_POSITIONS', '2'))
        self.POSITION_SIZE_PCT = float(os.getenv('POSITION_SIZE_PCT', '0.1'))  # 10% per position
        self.MIN_CASH_PCT = float(os.getenv('MIN_CASH_PCT', '0.1'))  # 10% minimum cash
        
        # Risk management
        self.STOP_LOSS_PCT = float(os.getenv('STOP_LOSS_PCT', '0.05'))  # 5% stop loss
        self.TAKE_PROFIT_PCT = float(os.getenv('TAKE_PROFIT_PCT', '0.15'))  # 15% take profit
        self.MAX_HOLD_DAYS = int(os.getenv('MAX_HOLD_DAYS', '30'))  # Max 30 days per trade
    
    def setup_backtesting_parameters(self):
        """Set up backtesting parameters."""
        self.BACKTEST_INIT_CASH = int(os.getenv('BACKTEST_INIT_CASH', '10000'))
        self.BACKTEST_MONTHS = int(os.getenv('BACKTEST_MONTHS', '6'))
        self.BACKTEST_START_DATE = datetime.now() - timedelta(days=self.BACKTEST_MONTHS * 30)
        
        # RSI optimization ranges
        self.RSI_PERIOD_RANGE = (
            int(os.getenv('RSI_PERIOD_START', '3')),
            int(os.getenv('RSI_PERIOD_STOP', '34')),
            int(os.getenv('RSI_PERIOD_STEP', '10'))
        )
        
        self.RSI_LOWER_RANGE = (
            int(os.getenv('RSI_LOWER_START', '20')),
            int(os.getenv('RSI_LOWER_STOP', '41')),
            int(os.getenv('RSI_LOWER_STEP', '5'))
        )
        
        self.RSI_UPPER_RANGE = (
            int(os.getenv('RSI_UPPER_START', '60')),
            int(os.getenv('RSI_UPPER_STOP', '85')),
            int(os.getenv('RSI_UPPER_STEP', '5'))
        )
    
    def setup_data_parameters(self):
        """Set up data filtering parameters."""
        self.MIN_VOLUME = int(os.getenv('MIN_VOLUME', '1000000'))  # 1M volume
        self.MIN_PRICE = float(os.getenv('MIN_PRICE', '15.0'))
        self.MAX_PRICE = float(os.getenv('MAX_PRICE', '200.0'))
        self.MIN_MARKET_CAP = float(os.getenv('MIN_MARKET_CAP', '1000000000'))  # 1B market cap
        
        # Google Cloud Storage
        self.GCS_BUCKET_NAME = os.getenv('GCS_BUCKET_NAME', 'trading-algo-data')
        
        # Rate limiting
        self.API_RATE_LIMIT_DELAY = float(os.getenv('API_RATE_LIMIT_DELAY', '0.2'))  # 200ms between calls
    
    def get_alpaca_config(self) -> Dict[str, str]:
        """Get Alpaca API configuration."""
        if self.PAPER_TRADE:
            return {
                'api_key': os.environ.get('ALPACA_PAPER_KEY', ''),
                'secret_key': os.environ.get('ALPACA_PAPER_SECRET', ''),
                'base_url': 'https://paper-api.alpaca.markets'
            }
        else:
            return {
                'api_key': os.environ.get('ALPACA_LIVE_KEY', ''),
                'secret_key': os.environ.get('ALPACA_LIVE_SECRET', ''),
                'base_url': 'https://api.alpaca.markets'
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for logging."""
        return {
            'paper_trade': self.PAPER_TRADE,
            'max_positions': self.MAX_POSITIONS,
            'max_new_positions_per_day': self.MAX_NEW_POSITIONS_PER_DAY,
            'position_size_pct': self.POSITION_SIZE_PCT,
            'min_cash_pct': self.MIN_CASH_PCT,
            'backtest_months': self.BACKTEST_MONTHS,
            'rsi_period_range': self.RSI_PERIOD_RANGE,
            'rsi_lower_range': self.RSI_LOWER_RANGE,
            'rsi_upper_range': self.RSI_UPPER_RANGE,
            'min_volume': self.MIN_VOLUME,
            'min_price': self.MIN_PRICE,
            'max_price': self.MAX_PRICE
        }
    
    def get_environment_path(self, base_path: str) -> str:
        """
        Get environment-specific path for cloud storage.
        
        Args:
            base_path: Base path (e.g., 'Backtests', 'Positions', 'Trades')
            
        Returns:
            Environment-specific path (e.g., 'dev/Backtests', 'qa/Positions', 'prod/trades')
        """
        return f"{self.ENVIRONMENT}/{base_path}"


# Global configuration instance
config = Config()
