"""Technical analysis indicator library (pure pandas/numpy functions).

Extracted from data_provider.py — indicators have no dependency on Alpaca or
any other app module, so they live standalone. Import as::

    from indicators import TechnicalIndicators

``data_provider`` re-exports ``TechnicalIndicators`` for backward compatibility.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Technical analysis indicators."""

    @staticmethod
    def calculate_rsi(data: pd.DataFrame, period: int = 14, price_col: Optional[str] = None) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).

        Args:
            data: DataFrame with price data
            period: RSI period
            price_col: Column name for price data (auto-detected if None)

        Returns:
            Series with RSI values
        """
        try:
            if len(data) < period + 1:
                return pd.Series(index=data.index, dtype=float)

            # Auto-detect price column if not specified
            if price_col is None:
                if 'close' in data.columns:
                    price_col = 'close'
                elif 'c' in data.columns:
                    price_col = 'c'
                else:
                    logger.error(
                        "No price column found. Expected 'close' or 'c' in data columns: %s", list(data.columns))
                    return pd.Series(index=data.index, dtype=float)

            # Ensure price column is numeric
            price_series = pd.to_numeric(data[price_col], errors='coerce')
            delta = price_series.diff()
            up = delta.copy()
            down = delta.copy()

            # Fix type issues with pandas operations - convert to numeric first
            up = pd.to_numeric(up, errors='coerce').where(
                pd.to_numeric(up, errors='coerce') > 0, 0)
            down = pd.to_numeric(down, errors='coerce').where(
                pd.to_numeric(down, errors='coerce') < 0, 0).abs()

            # Use exponential moving average
            rUp = up.ewm(com=period - 1, adjust=False).mean()
            rDown = down.ewm(com=period - 1, adjust=False).mean()

            # Avoid division by zero
            rDown = rDown.replace(0, np.nan)
            rs = rUp / rDown
            rsi = 100 - (100 / (1 + rs))

            return rsi.fillna(50)  # Fill NaN with neutral RSI value

        except Exception as e:
            logger.error("Error calculating RSI: %s", e)
            return pd.Series(index=data.index, dtype=float)

    @staticmethod
    def calculate_sma(data: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return data.rolling(window=period, min_periods=1).mean()

    @staticmethod
    def calculate_ema(data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return data.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_moving_average(data: pd.Series, window: int) -> pd.Series:
        """Calculate Simple Moving Average - alias for compatibility."""
        return TechnicalIndicators.calculate_sma(data, window)

    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.

        Args:
            data: Price series
            window: Period for moving average
            num_std: Number of standard deviations for bands

        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        middle = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        return upper, middle, lower

    @staticmethod
    def calculate_macd(data: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Args:
            data: Price series
            fast_period: Fast EMA period
            slow_period: Slow EMA period  
            signal_period: Signal line EMA period

        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        ema_fast = TechnicalIndicators.calculate_ema(data, fast_period)
        ema_slow = TechnicalIndicators.calculate_ema(data, slow_period)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.calculate_ema(
            macd_line, signal_period)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator.

        Args:
            high: High price series
            low: Low price series
            close: Close price series
            k_period: Period for %K calculation
            d_period: Period for %D (signal line) calculation

        Returns:
            Tuple of (%K, %D)
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()

        return k_percent, d_percent


# Global data provider instance
