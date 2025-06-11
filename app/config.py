"""
Configuration module for the trading algorithm.
Contains all parameters and environment variables.
"""
import os
import json
from datetime import datetime, timedelta, date
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class for trading algorithm parameters."""
    
    def __init__(self):
        self.load_environment_variables()
        self.load_json_config()
        self.setup_data_parameters()
    
    def load_environment_variables(self):
        """Load API keys and credentials from environment variables."""
        # Alpaca credentials are read directly from environment variables
        # Set these environment variables:
        # - ALPACA_DEV_PAPER_KEY: Your dev paper trading API key
        # - ALPACA_DEV_PAPER_SECRET: Your dev paper trading secret key
        # - ALPACA_QA_PAPER_KEY: Your qa paper trading API key
        # - ALPACA_QA_PAPER_SECRET: Your qa paper trading secret key
        # - ALPACA_LIVE_KEY: Your live trading API key (for prod only)
        # - ALPACA_LIVE_SECRET: Your live trading secret key (for prod only)
        # - GOOGLE_APPLICATION_CREDENTIALS: Path to GCS service account JSON (optional)
        # - ENVIRONMENT: Environment setting (dev, qa, prod) - defaults to 'dev'
        
        # Environment setting (dev, qa, prod)
        self.ENVIRONMENT = os.getenv('ENVIRONMENT', 'dev').lower()
        if self.ENVIRONMENT not in ['dev', 'qa', 'prod']:
            print(f"Warning: Invalid ENVIRONMENT value '{self.ENVIRONMENT}'. Must be 'dev', 'qa', or 'prod'. Defaulting to 'dev'.")
            self.ENVIRONMENT = 'dev'
        
        # Check for required environment variables based on environment
        if self.ENVIRONMENT == 'dev':
            required_vars = ['ALPACA_DEV_PAPER_KEY', 'ALPACA_DEV_PAPER_SECRET']
        elif self.ENVIRONMENT == 'qa':
            required_vars = ['ALPACA_QA_PAPER_KEY', 'ALPACA_QA_PAPER_SECRET']
        elif self.ENVIRONMENT == 'prod':
            required_vars = ['ALPACA_LIVE_KEY', 'ALPACA_LIVE_SECRET']
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            print(f"Warning: Missing required environment variables for {self.ENVIRONMENT} environment: {missing_vars}")
            print("Set these environment variables for the algorithm to work properly.")
        
        # Optional variables
        optional_vars = ['GOOGLE_APPLICATION_CREDENTIALS']
        for var in optional_vars:
            if not os.getenv(var):
                print(f"Info: Optional environment variable {var} not set.")
        
        print(f"Info: Running in '{self.ENVIRONMENT}' environment")
    
    def load_json_config(self):
        """Load configuration from environment-specific JSON file."""
        config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config')
        config_file = os.path.join(config_dir, f'{self.ENVIRONMENT}.json')
        
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Trading parameters
            trading = config_data.get('trading', {})
            self.PAPER_TRADE = trading.get('paper_trade', True)
            self.MAX_POSITIONS = trading.get('max_positions', 10)
            self.MAX_NEW_POSITIONS_PER_DAY = trading.get('max_new_positions', 2)
            self.POSITION_SIZE_PCT = trading.get('position_size_pct', 0.1)
            self.MIN_CASH_PCT = trading.get('min_cash_pct', 0.1)
            self.STOP_LOSS_PCT = trading.get('stop_loss_pct', 0.05)
            self.TAKE_PROFIT_PCT = trading.get('take_profit_pct', 0.15)
            self.MAX_HOLD_DAYS = trading.get('max_hold_days', 30)
            self.MIN_WIN_RATE = trading.get('min_win_rate', 0.7)
            
            # Backtesting parameters
            backtesting = config_data.get('backtesting', {})
            self.BACKTEST_INIT_CASH = backtesting.get('init_cash', 10000)
            self.BACKTEST_MONTHS = backtesting.get('months', 6)
            self.BACKTEST_START_DATE = datetime.now() - timedelta(days=self.BACKTEST_MONTHS * 30)
            
            # RSI optimization ranges
            rsi_opt = config_data.get('rsi_optimization', {})
            period_range = rsi_opt.get('period_range', {})
            lower_range = rsi_opt.get('lower_range', {})
            upper_range = rsi_opt.get('upper_range', {})
            
            self.RSI_PERIOD_RANGE = (
                period_range.get('start', 3),
                period_range.get('stop', 34),
                period_range.get('step', 10)
            )
            
            self.RSI_LOWER_RANGE = (
                lower_range.get('start', 20),
                lower_range.get('stop', 41),
                lower_range.get('step', 5)
            )
            
            self.RSI_UPPER_RANGE = (
                upper_range.get('start', 60),
                upper_range.get('stop', 85),
                upper_range.get('step', 5)
            )
            
            # Data filtering parameters
            data_filtering = config_data.get('data_filtering', {})
            self.MIN_VOLUME = data_filtering.get('min_volume', 1000000)
            self.MIN_PRICE = data_filtering.get('min_price', 15.0)
            self.MAX_PRICE = data_filtering.get('max_price', 200.0)
            self.MIN_MARKET_CAP = data_filtering.get('min_market_cap', 1000000000)
            
            # API parameters
            api = config_data.get('api', {})
            self.API_RATE_LIMIT_DELAY = api.get('rate_limit_delay', 0.1)
            
            print(f"Info: Loaded configuration from {config_file}")
            
        except FileNotFoundError:
            print(f"Warning: Configuration file {config_file} not found. Using default values.")
            self.setup_trading_parameters()
            self.setup_backtesting_parameters()
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in {config_file}: {e}. Using default values.")
            self.setup_trading_parameters()
            self.setup_backtesting_parameters()
    
    def setup_trading_parameters(self):
        """Set up trading-related parameters (fallback method)."""
        # Trading mode
        self.PAPER_TRADE = True
        
        # Portfolio management
        self.MAX_POSITIONS = 10
        self.MAX_NEW_POSITIONS_PER_DAY = 2
        self.POSITION_SIZE_PCT = 0.1  # 10% per position
        self.MIN_CASH_PCT = 0.1  # 10% minimum cash
        
        # Risk management
        self.STOP_LOSS_PCT = 0.05  # 5% stop loss
        self.TAKE_PROFIT_PCT = 0.15  # 15% take profit
        self.MAX_HOLD_DAYS = 30  # Max 30 days per trade
        self.MIN_WIN_RATE = 0.7  # Minimum win rate of 70%
    
    def setup_backtesting_parameters(self):
        """Set up backtesting parameters (fallback method)."""
        self.BACKTEST_INIT_CASH = 10000
        self.BACKTEST_MONTHS = 6
        self.BACKTEST_START_DATE = datetime.now() - timedelta(days=self.BACKTEST_MONTHS * 30)
        
        # RSI optimization ranges
        self.RSI_PERIOD_RANGE = (3, 34, 10)
        self.RSI_LOWER_RANGE = (20, 41, 5)
        self.RSI_UPPER_RANGE = (60, 85, 5)
    
    def setup_data_parameters(self):
        """Set up data filtering parameters (fallback handled in load_json_config)."""
        # Google Cloud Storage
        self.GCS_BUCKET_NAME = os.getenv('GCS_BUCKET_NAME', 'trading-algo-data')
    
    def get_alpaca_config(self) -> Dict[str, str]:
        """
        Get Alpaca API configuration based on environment.
        
        Environment-specific key mapping:
        - dev: Uses ALPACA_DEV_PAPER_KEY/SECRET (always paper trading)
        - qa: Uses ALPACA_QA_PAPER_KEY/SECRET (always paper trading)  
        - prod: Uses ALPACA_LIVE_KEY/SECRET for live trading, or QA paper for safety
        
        Returns:
            Dictionary with api_key, secret_key, and base_url
        """
        if self.ENVIRONMENT == 'dev':
            # Dev environment always uses paper trading
            return {
                'api_key': os.environ.get('ALPACA_DEV_PAPER_KEY', ''),
                'secret_key': os.environ.get('ALPACA_DEV_PAPER_SECRET', ''),
                'base_url': 'https://paper-api.alpaca.markets'
            }
        elif self.ENVIRONMENT == 'qa':
            # QA environment always uses paper trading
            return {
                'api_key': os.environ.get('ALPACA_QA_PAPER_KEY', ''),
                'secret_key': os.environ.get('ALPACA_QA_PAPER_SECRET', ''),
                'base_url': 'https://paper-api.alpaca.markets'
            }
        elif self.ENVIRONMENT == 'prod':
            # Prod environment can use either paper or live trading based on PAPER_TRADE setting
            if self.PAPER_TRADE:
                # Use QA paper account for prod paper trading as a safety measure
                return {
                    'api_key': os.environ.get('ALPACA_QA_PAPER_KEY', ''),
                    'secret_key': os.environ.get('ALPACA_QA_PAPER_SECRET', ''),
                    'base_url': 'https://paper-api.alpaca.markets'
                }
            else:
                # Use live trading account
                return {
                    'api_key': os.environ.get('ALPACA_LIVE_KEY', ''),
                    'secret_key': os.environ.get('ALPACA_LIVE_SECRET', ''),
                    'base_url': 'https://api.alpaca.markets'
                }
        else:
            # Fallback to dev paper account
            return {
                'api_key': os.environ.get('ALPACA_DEV_PAPER_KEY', ''),
                'secret_key': os.environ.get('ALPACA_DEV_PAPER_SECRET', ''),
                'base_url': 'https://paper-api.alpaca.markets'
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
