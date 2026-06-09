#!/usr/bin/env python3
"""Unit tests for the current utils module API."""
import os
import sys
import unittest
from datetime import datetime
from unittest.mock import Mock, patch

import pandas as pd

# Add app path before importing module-under-test.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from utils import (  # noqa: E402
    DataValidator,
    PerformanceMetrics,
    ProgressIndicator,
    RiskManager,
    TradingCalendar,
    format_currency,
    format_percentage,
    is_trading_day,
    parse_dt,
    setup_logging,
)


class TestUtilityFunctions(unittest.TestCase):
    """Tests for module-level utility functions."""

    @patch('utils.Path.mkdir')
    @patch('utils.logging.basicConfig')
    def test_setup_logging(self, mock_basic_config, mock_mkdir):
        logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        setup_logging('DEBUG')

        mock_mkdir.assert_called_once_with(exist_ok=True)
        mock_basic_config.assert_called_once()

    @patch('utils.importlib.import_module')
    def test_is_trading_day_weekday(self, mock_import_module):
        mock_holidays_module = Mock()
        mock_holidays_module.country_holidays.return_value = set()
        mock_import_module.return_value = mock_holidays_module

        result = is_trading_day(datetime(2025, 6, 11))  # Wednesday
        self.assertTrue(result)

    @patch('utils.importlib.import_module')
    def test_is_trading_day_weekend(self, mock_import_module):
        mock_holidays_module = Mock()
        mock_holidays_module.country_holidays.return_value = set()
        mock_import_module.return_value = mock_holidays_module

        result = is_trading_day(datetime(2025, 6, 14))  # Saturday
        self.assertFalse(result)

    @patch('utils.importlib.import_module')
    def test_is_trading_day_holiday(self, mock_import_module):
        mock_holidays_module = Mock()
        mock_holidays_module.country_holidays.return_value = {
            datetime(2025, 7, 4).date()}
        mock_import_module.return_value = mock_holidays_module

        result = is_trading_day(datetime(2025, 7, 4))
        self.assertFalse(result)

    def test_format_currency(self):
        self.assertEqual(format_currency(1234.56), "$1,234.56")
        # Current implementation uses "$-1,234.56" style.
        self.assertEqual(format_currency(-1234.56), "$-1,234.56")
        self.assertEqual(format_currency(0), "$0.00")

    def test_format_percentage(self):
        self.assertEqual(format_percentage(0.1234), "12.34%")
        self.assertEqual(format_percentage(-0.1234), "-12.34%")
        self.assertEqual(format_percentage(0), "0.00%")


class TestTradingCalendar(unittest.TestCase):
    """Tests for TradingCalendar class."""

    def setUp(self):
        self.calendar = TradingCalendar()

    @patch('utils.is_trading_day', return_value=True)
    def test_is_market_open_true(self, _):
        dt = datetime(2025, 6, 11, 10, 0, 0)  # Wednesday 10:00 ET-naive
        self.assertTrue(self.calendar.is_market_open(dt))

    @patch('utils.is_trading_day', return_value=True)
    def test_is_market_open_false(self, _):
        dt = datetime(2025, 6, 11, 8, 0, 0)  # Before open
        self.assertFalse(self.calendar.is_market_open(dt))


class TestPerformanceMetrics(unittest.TestCase):
    """Tests for PerformanceMetrics helpers."""

    def test_calculate_sharpe_ratio(self):
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        sharpe = PerformanceMetrics.calculate_sharpe_ratio(returns)
        self.assertIsInstance(sharpe, float)

    def test_calculate_max_drawdown(self):
        values = pd.Series([100, 110, 105, 120, 90, 95])
        dd = PerformanceMetrics.calculate_max_drawdown(values)
        self.assertGreaterEqual(dd, 0.0)


class TestDataValidator(unittest.TestCase):
    """Tests for data quality helpers."""

    def test_validate_price_data_valid(self):
        df = pd.DataFrame({
            'o': [100, 101],
            'h': [105, 106],
            'l': [99, 100],
            'c': [102, 104],
            'v': [1000, 1200],
        })
        self.assertTrue(DataValidator.validate_price_data(df))

    def test_validate_price_data_invalid_schema(self):
        df = pd.DataFrame({'close': [1, 2, 3]})
        self.assertFalse(DataValidator.validate_price_data(df))

    def test_detect_outliers(self):
        s = pd.Series([1, 1, 1, 100])
        outliers = DataValidator.detect_outliers(s, threshold=1.0)
        self.assertEqual(len(outliers), len(s))
        self.assertTrue(outliers.iloc[-1])


class TestRiskManager(unittest.TestCase):
    """Tests for RiskManager helpers."""

    def test_calculate_position_size(self):
        size = RiskManager.calculate_position_size(
            account_value=100000,
            risk_per_trade=0.02,
            entry_price=100,
            stop_loss_price=95,
        )
        self.assertEqual(size, 400)

    def test_check_correlation(self):
        r1 = pd.Series([0.01, 0.02, -0.01, 0.03])
        r2 = pd.Series([0.02, 0.01, -0.02, 0.04])
        corr = RiskManager.check_correlation(r1, r2)
        self.assertIsInstance(corr, float)
        self.assertLessEqual(corr, 1.0)
        self.assertGreaterEqual(corr, -1.0)


class TestProgressIndicator(unittest.TestCase):
    """Tests for ProgressIndicator class."""

    def test_progress_indicator_init(self):
        progress = ProgressIndicator(total=100, description="Test")
        self.assertEqual(progress.total, 100)
        self.assertEqual(progress.description, "Test")
        self.assertEqual(progress.current, 0)

    @patch('utils.sys.stdout.write')
    @patch('utils.sys.stdout.flush')
    def test_progress_indicator_update(self, mock_flush, mock_write):
        progress = ProgressIndicator(total=100, description="Test")
        progress.update(25)
        self.assertEqual(progress.current, 25)
        mock_write.assert_called()
        mock_flush.assert_called()

    @patch('utils.sys.stdout.write')
    @patch('utils.sys.stdout.flush')
    def test_progress_indicator_finish(self, mock_flush, mock_write):
        progress = ProgressIndicator(total=100, description="Test")
        progress.finish("Done")
        mock_write.assert_called()
        mock_flush.assert_called()


class TestParseDt(unittest.TestCase):
    """Tests for parse_dt — ensures datetime subtraction never fails on CSV strings."""

    def test_parse_dt_datetime_passthrough(self):
        """datetime objects should pass through unchanged."""
        dt = datetime(2025, 6, 14, 10, 30)
        result = parse_dt(dt)
        self.assertEqual(result, dt)
        self.assertIsInstance(result, datetime)

    def test_parse_dt_pandas_timestamp(self):
        """pd.Timestamp should be converted to datetime."""
        ts = pd.Timestamp("2025-06-14 10:30:00")
        result = parse_dt(ts)
        self.assertIsInstance(result, datetime)
        self.assertEqual(result, datetime(2025, 6, 14, 10, 30))

    def test_parse_dt_string(self):
        """String from CSV should be converted to datetime."""
        result = parse_dt("2025-06-14")
        self.assertIsInstance(result, datetime)
        self.assertEqual(result, datetime(2025, 6, 14))

    def test_parse_dt_string_with_time(self):
        """String with time portion should parse correctly."""
        result = parse_dt("2025-06-14 10:30:00")
        self.assertIsInstance(result, datetime)
        self.assertEqual(result, datetime(2025, 6, 14, 10, 30))

    def test_parse_dt_none_returns_default(self):
        """None should return the default value."""
        self.assertIsNone(parse_dt(None))
        self.assertEqual(parse_dt(None, default="fallback"), "fallback")

    def test_parse_dt_invalid_string_returns_default(self):
        """Invalid date string should return default and not raise."""
        result = parse_dt("not-a-date", default=None)
        self.assertIsNone(result)

    def test_parse_dt_string_allows_subtraction(self):
        """Regression: verify that a parsed string date can be subtracted from datetime."""
        now = datetime(2025, 6, 14, 12, 0)
        entry_date = parse_dt("2025-06-07")
        self.assertIsNotNone(entry_date)
        assert entry_date is not None  # narrow type for type checker
        days_held = (now - entry_date).days
        self.assertEqual(days_held, 7)

    def test_parse_dt_float_timestamp(self):
        """Unix-style float timestamps should parse correctly."""
        result = parse_dt(1718312400.0)  # 2024-06-14 midnight UTC
        self.assertIsInstance(result, datetime)


if __name__ == '__main__':
    unittest.main()
