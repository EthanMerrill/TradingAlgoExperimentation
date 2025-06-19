#!/usr/bin/env python3
"""
Unit tests for the utils module.
"""
from utils import (
    setup_logging, is_trading_day, is_market_hours, get_market_hours,
    calculate_rsi, calculate_portfolio_metrics, round_to_nearest_cent,
    format_currency, format_percentage, validate_symbol, ProgressIndicator
)
import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from datetime import datetime, date
import pandas as pd
import numpy as np
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""

    @patch('utils.Path.mkdir')
    @patch('utils.logging.basicConfig')
    def test_setup_logging(self, mock_basic_config, mock_mkdir):
        """Test logging setup."""
        setup_logging('DEBUG')

        mock_mkdir.assert_called_once_with(exist_ok=True)
        mock_basic_config.assert_called_once()

    @patch('utils.holidays.UnitedStates')
    def test_is_trading_day_weekday(self, mock_holidays):
        """Test is_trading_day for a regular weekday."""
        mock_holidays.return_value = {}

        # Test with a Wednesday (not a holiday)
        test_date = datetime(2025, 6, 11)  # Wednesday
        result = is_trading_day(test_date)

        self.assertTrue(result)

    @patch('utils.holidays.UnitedStates')
    def test_is_trading_day_weekend(self, mock_holidays):
        """Test is_trading_day for a weekend."""
        mock_holidays.return_value = {}

        # Test with a Saturday
        test_date = datetime(2025, 6, 14)  # Saturday
        result = is_trading_day(test_date)

        self.assertFalse(result)

    @patch('utils.holidays.UnitedStates')
    def test_is_trading_day_holiday(self, mock_holidays):
        """Test is_trading_day for a holiday."""
        mock_holidays.return_value = {date(2025, 7, 4): "Independence Day"}

        # Test with July 4th (Independence Day)
        test_date = datetime(2025, 7, 4)
        result = is_trading_day(test_date)

        self.assertFalse(result)

    @patch('utils.datetime')
    def test_is_market_hours_during_trading(self, mock_datetime):
        """Test is_market_hours during trading hours."""
        # Mock current time to be 2 PM ET on a Wednesday
        mock_datetime.now.return_value = datetime(2025, 6, 11, 14, 0, 0)

        with patch('utils.is_trading_day', return_value=True):
            result = is_market_hours()

            self.assertTrue(result)

    @patch('utils.datetime')
    def test_is_market_hours_outside_trading(self, mock_datetime):
        """Test is_market_hours outside trading hours."""
        # Mock current time to be 8 AM ET on a Wednesday (before market open)
        mock_datetime.now.return_value = datetime(2025, 6, 11, 8, 0, 0)

        with patch('utils.is_trading_day', return_value=True):
            result = is_market_hours()

            self.assertFalse(result)

    def test_get_market_hours(self):
        """Test get_market_hours function."""
        test_date = datetime(2025, 6, 11)
        market_open, market_close = get_market_hours(test_date)

        self.assertEqual(market_open.hour, 9)
        self.assertEqual(market_open.minute, 30)
        self.assertEqual(market_close.hour, 16)
        self.assertEqual(market_close.minute, 0)

    def test_calculate_rsi_with_valid_data(self):
        """Test RSI calculation with valid price data."""
        # Create sample price data
        prices = pd.Series([100, 105, 103, 108, 110, 107,
                           112, 115, 113, 118, 120, 117, 122, 125, 123])

        rsi = calculate_rsi(prices, period=14)

        self.assertIsInstance(rsi, pd.Series)
        self.assertEqual(len(rsi), len(prices))
        # RSI should be between 0 and 100
        self.assertTrue(all(0 <= val <= 100 for val in rsi.dropna()))

    def test_calculate_rsi_insufficient_data(self):
        """Test RSI calculation with insufficient data."""
        prices = pd.Series([100, 105, 103])  # Only 3 data points

        rsi = calculate_rsi(prices, period=14)

        # Should return NaN for insufficient data
        self.assertTrue(rsi.isna().all())

    def test_calculate_portfolio_metrics(self):
        """Test portfolio metrics calculation."""
        returns = pd.Series(
            [0.01, -0.02, 0.03, -0.01, 0.02, 0.01, -0.03, 0.04])

        metrics = calculate_portfolio_metrics(returns)

        self.assertIn('total_return', metrics)
        self.assertIn('volatility', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown', metrics)
        self.assertIn('win_rate', metrics)

    def test_round_to_nearest_cent(self):
        """Test rounding to nearest cent."""
        self.assertEqual(round_to_nearest_cent(1.234), 1.23)
        self.assertEqual(round_to_nearest_cent(1.236), 1.24)
        self.assertEqual(round_to_nearest_cent(
            1.235), 1.24)  # Banker's rounding

    def test_format_currency(self):
        """Test currency formatting."""
        self.assertEqual(format_currency(1234.56), "$1,234.56")
        self.assertEqual(format_currency(-1234.56), "-$1,234.56")
        self.assertEqual(format_currency(0), "$0.00")

    def test_format_percentage(self):
        """Test percentage formatting."""
        self.assertEqual(format_percentage(0.1234), "12.34%")
        self.assertEqual(format_percentage(-0.1234), "-12.34%")
        self.assertEqual(format_percentage(0), "0.00%")

    def test_validate_symbol_valid(self):
        """Test symbol validation with valid symbols."""
        self.assertTrue(validate_symbol("AAPL"))
        self.assertTrue(validate_symbol("TSLA"))
        self.assertTrue(validate_symbol("BRK.A"))

    def test_validate_symbol_invalid(self):
        """Test symbol validation with invalid symbols."""
        self.assertFalse(validate_symbol(""))
        self.assertFalse(validate_symbol("invalid_symbol"))
        self.assertFalse(validate_symbol("123"))
        self.assertFalse(validate_symbol("a" * 10))  # Too long


class TestProgressIndicator(unittest.TestCase):
    """Test cases for ProgressIndicator class."""

    def test_progress_indicator_init(self):
        """Test ProgressIndicator initialization."""
        progress = ProgressIndicator(total=100, description="Test")

        self.assertEqual(progress.total, 100)
        self.assertEqual(progress.description, "Test")
        self.assertEqual(progress.current, 0)

    @patch('utils.sys.stdout.write')
    @patch('utils.sys.stdout.flush')
    def test_progress_indicator_update(self, mock_flush, mock_write):
        """Test ProgressIndicator update method."""
        progress = ProgressIndicator(total=100, description="Test")

        progress.update(25)

        self.assertEqual(progress.current, 25)
        mock_write.assert_called()
        mock_flush.assert_called()

    @patch('utils.sys.stdout.write')
    @patch('utils.sys.stdout.flush')
    def test_progress_indicator_complete(self, mock_flush, mock_write):
        """Test ProgressIndicator completion."""
        progress = ProgressIndicator(total=100, description="Test")

        progress.update(100)

        self.assertEqual(progress.current, 100)
        mock_write.assert_called()
        mock_flush.assert_called()

    def test_progress_indicator_context_manager(self):
        """Test ProgressIndicator as context manager."""
        with patch('utils.sys.stdout.write'), patch('utils.sys.stdout.flush'):
            with ProgressIndicator(total=100, description="Test") as progress:
                self.assertIsNotNone(progress)
                progress.update(50)
                self.assertEqual(progress.current, 50)


if __name__ == '__main__':
    unittest.main()
