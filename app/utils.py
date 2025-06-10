"""
Utility functions and helper classes for the trading algorithm.
"""
import logging
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Optional, List
import pytz
import holidays
from pathlib import Path


def setup_logging(level: str = 'INFO') -> None:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logs directory if it doesn't exist
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=[
            logging.FileHandler(logs_dir / f'trading_algo_{datetime.now().strftime("%Y%m%d")}.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set third-party loggers to WARNING to reduce noise
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('google').setLevel(logging.WARNING)
    logging.getLogger('alpaca').setLevel(logging.WARNING)


def is_trading_day(date: datetime = None) -> bool:
    """
    Check if a given date is a trading day (market open).
    
    Args:
        date: Date to check (defaults to today)
        
    Returns:
        True if it's a trading day
    """
    if date is None:
        date = datetime.now()
    
    # Check if it's a weekend
    if date.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False
    
    # Check if it's a US market holiday
    us_holidays = holidays.US(years=date.year)
    
    # Additional market-specific holidays
    market_holidays = [
        # Add any additional market holidays that aren't in the standard US holidays
    ]
    
    if date.date() in us_holidays:
        return False
    
    for holiday_date in market_holidays:
        if date.date() == holiday_date:
            return False
    
    return True


class TradingCalendar:
    """Helper class for trading calendar operations."""
    
    def __init__(self):
        self.us_eastern = pytz.timezone('US/Eastern')
        self.market_open_time = (9, 30)  # 9:30 AM ET
        self.market_close_time = (16, 0)  # 4:00 PM ET
    
    def is_trading_day(self, date: datetime = None) -> bool:
        """Check if it's a trading day."""
        return is_trading_day(date)
    
    def is_market_open(self, dt: datetime = None) -> bool:
        """
        Check if the market is currently open.
        
        Args:
            dt: Datetime to check (defaults to now)
            
        Returns:
            True if market is open
        """
        if dt is None:
            dt = datetime.now()
        
        # Convert to Eastern Time
        if dt.tzinfo is None:
            dt = self.us_eastern.localize(dt)
        else:
            dt = dt.astimezone(self.us_eastern)
        
        # Check if it's a trading day
        if not self.is_trading_day(dt):
            return False
        
        # Check if it's during trading hours
        market_open = dt.replace(hour=self.market_open_time[0], 
                                minute=self.market_open_time[1], 
                                second=0, microsecond=0)
        market_close = dt.replace(hour=self.market_close_time[0], 
                                 minute=self.market_close_time[1], 
                                 second=0, microsecond=0)
        
        return market_open <= dt <= market_close
    
    def next_trading_day(self, date: datetime = None) -> datetime:
        """Get the next trading day."""
        if date is None:
            date = datetime.now()
        
        next_day = date + timedelta(days=1)
        while not self.is_trading_day(next_day):
            next_day += timedelta(days=1)
        
        return next_day
    
    def previous_trading_day(self, date: datetime = None) -> datetime:
        """Get the previous trading day."""
        if date is None:
            date = datetime.now()
        
        prev_day = date - timedelta(days=1)
        while not self.is_trading_day(prev_day):
            prev_day -= timedelta(days=1)
        
        return prev_day


class PerformanceMetrics:
    """Calculate various performance metrics for trading strategies."""
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        return np.sqrt(252) * excess_returns.mean() / returns.std()
    
    @staticmethod
    def calculate_max_drawdown(values: pd.Series) -> float:
        """Calculate maximum drawdown."""
        peak = values.expanding().max()
        drawdown = (values - peak) / peak
        return abs(drawdown.min())
    
    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series, values: pd.Series) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)."""
        annual_return = (1 + returns.mean()) ** 252 - 1
        max_dd = PerformanceMetrics.calculate_max_drawdown(values)
        
        if max_dd == 0:
            return float('inf') if annual_return > 0 else 0
        
        return annual_return / max_dd
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (uses downside deviation instead of total volatility)."""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / 252)
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return float('inf') if excess_returns.mean() > 0 else 0
        
        return np.sqrt(252) * excess_returns.mean() / downside_returns.std()


class DataValidator:
    """Validate data quality and integrity."""
    
    @staticmethod
    def validate_price_data(df: pd.DataFrame) -> bool:
        """
        Validate price data for basic quality checks.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            True if data passes validation
        """
        required_columns = ['o', 'h', 'l', 'c', 'v']
        
        # Check if required columns exist
        if not all(col in df.columns for col in required_columns):
            return False
        
        # Check for reasonable price relationships
        if not (df['l'] <= df['c']).all() or not (df['c'] <= df['h']).all():
            return False
        
        # Check for positive prices
        if not (df[['o', 'h', 'l', 'c']] > 0).all().all():
            return False
        
        # Check for reasonable volume
        if not (df['v'] >= 0).all():
            return False
        
        return True
    
    @staticmethod
    def detect_outliers(series: pd.Series, threshold: float = 3.0) -> pd.Series:
        """
        Detect outliers using z-score method.
        
        Args:
            series: Data series
            threshold: Z-score threshold for outliers
            
        Returns:
            Boolean series indicating outliers
        """
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > threshold
    
    @staticmethod
    def clean_price_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean price data by removing obvious errors.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Remove rows where high < low (impossible)
        df_clean = df_clean[df_clean['h'] >= df_clean['l']]
        
        # Remove rows with zero or negative prices
        df_clean = df_clean[df_clean[['o', 'h', 'l', 'c']].min(axis=1) > 0]
        
        # Remove extreme price movements (>50% in one day)
        daily_change = df_clean['c'].pct_change().abs()
        df_clean = df_clean[daily_change <= 0.5]
        
        return df_clean


class RiskManager:
    """Risk management utilities."""
    
    @staticmethod
    def calculate_position_size(
        account_value: float,
        risk_per_trade: float,
        entry_price: float,
        stop_loss_price: float
    ) -> int:
        """
        Calculate position size based on risk management rules.
        
        Args:
            account_value: Total account value
            risk_per_trade: Risk per trade as percentage (e.g., 0.02 for 2%)
            entry_price: Entry price per share
            stop_loss_price: Stop loss price per share
            
        Returns:
            Number of shares to buy
        """
        if entry_price <= 0 or stop_loss_price <= 0:
            return 0
        
        risk_amount = account_value * risk_per_trade
        risk_per_share = abs(entry_price - stop_loss_price)
        
        if risk_per_share == 0:
            return 0
        
        position_size = int(risk_amount / risk_per_share)
        return max(0, position_size)
    
    @staticmethod
    def check_correlation(returns1: pd.Series, returns2: pd.Series) -> float:
        """
        Check correlation between two return series.
        
        Args:
            returns1: First return series
            returns2: Second return series
            
        Returns:
            Correlation coefficient
        """
        if len(returns1) == 0 or len(returns2) == 0:
            return 0.0
        
        # Align series by index
        aligned_data = pd.concat([returns1, returns2], axis=1).dropna()
        
        if len(aligned_data) < 2:
            return 0.0
        
        return aligned_data.corr().iloc[0, 1]


def format_currency(amount: float) -> str:
    """Format a number as currency."""
    return f"${amount:,.2f}"


def format_percentage(value: float) -> str:
    """Format a decimal as percentage."""
    return f"{value:.2%}"


def calculate_business_days(start_date: datetime, end_date: datetime) -> int:
    """Calculate number of business days between two dates."""
    return pd.bdate_range(start_date, end_date).size


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if division by zero."""
    if denominator == 0:
        return default
    return numerator / denominator
