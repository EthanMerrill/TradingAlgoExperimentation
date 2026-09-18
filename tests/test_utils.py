#!/usr/bin/env python3
"""Unit tests for the current utils module API."""
import os
import sys
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import pandas as pd

# Add app path before importing module-under-test.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from utils import (  # noqa: E402
    PerformanceMetrics,
    ProgressIndicator,
    is_trading_day,
    parse_dt,
    setup_logging,
    utc_now,
)


class TestUtilityFunctions(unittest.TestCase):
    """Tests for module-level utility functions."""

    @patch('utils.logging_.Path.mkdir')
    @patch('utils.logging_.logging.basicConfig')
    def test_setup_logging(self, mock_basic_config, mock_mkdir):
        logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        setup_logging('DEBUG')

        mock_mkdir.assert_called_once_with(exist_ok=True)
        mock_basic_config.assert_called_once()

    @patch('utils.datetime_.importlib.import_module')
    def test_is_trading_day_weekday(self, mock_import_module):
        mock_holidays_module = Mock()
        mock_holidays_module.country_holidays.return_value = set()
        mock_import_module.return_value = mock_holidays_module

        result = is_trading_day(datetime(2025, 6, 11))  # Wednesday
        self.assertTrue(result)

    @patch('utils.datetime_.importlib.import_module')
    def test_is_trading_day_weekend(self, mock_import_module):
        mock_holidays_module = Mock()
        mock_holidays_module.country_holidays.return_value = set()
        mock_import_module.return_value = mock_holidays_module

        result = is_trading_day(datetime(2025, 6, 14))  # Saturday
        self.assertFalse(result)

    @patch('utils.datetime_.importlib.import_module')
    def test_is_trading_day_holiday(self, mock_import_module):
        mock_holidays_module = Mock()
        mock_holidays_module.country_holidays.return_value = {
            datetime(2025, 7, 4).date()}
        mock_import_module.return_value = mock_holidays_module

        result = is_trading_day(datetime(2025, 7, 4))
        self.assertFalse(result)


class TestPerformanceMetrics(unittest.TestCase):
    """Tests for PerformanceMetrics helpers."""

    def test_calculate_max_drawdown(self):
        values = pd.Series([100, 110, 105, 120, 90, 95])
        dd = PerformanceMetrics.calculate_max_drawdown(values)
        self.assertGreaterEqual(dd, 0.0)


class TestProgressIndicator(unittest.TestCase):
    """Tests for ProgressIndicator class."""

    def test_progress_indicator_init(self):
        progress = ProgressIndicator(total=100, description="Test")
        self.assertEqual(progress.total, 100)
        self.assertEqual(progress.description, "Test")
        self.assertEqual(progress.current, 0)

    @patch('utils.progress.sys.stdout.write')
    @patch('utils.progress.sys.stdout.flush')
    def test_progress_indicator_update(self, mock_flush, mock_write):
        progress = ProgressIndicator(total=100, description="Test")
        progress.update(25)
        self.assertEqual(progress.current, 25)
        mock_write.assert_called()
        mock_flush.assert_called()

    @patch('utils.progress.sys.stdout.write')
    @patch('utils.progress.sys.stdout.flush')
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


class TestUtcNow(unittest.TestCase):
    """`utc_now` exists because `datetime.now()` is naive LOCAL time.

    Persistence treats naive values as UTC wall-clock, so a local timestamp
    lands in the database shifted by the machine's UTC offset — which produced
    order rows whose `submitted_at` sat 4 hours behind their `created_at`.
    """

    def test_returns_aware_utc_datetime(self):
        now = utc_now()
        self.assertIsInstance(now, datetime)
        self.assertIsNotNone(now.tzinfo)
        self.assertEqual(now.utcoffset(), timedelta(0))

    def test_matches_utc_now_within_tolerance(self):
        before = datetime.now(timezone.utc)
        value = utc_now()
        after = datetime.now(timezone.utc)
        self.assertLessEqual(before, value)
        self.assertLessEqual(value, after)

    def test_differs_from_naive_local_now(self):
        """Sanity: the helper must not be a naive local timestamp.

        Skipped in UTC zones, where the two are indistinguishable in wall clock.
        """
        if datetime.now().astimezone().utcoffset() == timedelta(0):
            self.skipTest("local timezone is UTC")
        # as a naive wall clock the UTC value must not equal local wall clock
        self.assertNotEqual(
            utc_now().replace(tzinfo=None).replace(microsecond=0),
            datetime.now().replace(microsecond=0),
        )

    def test_is_accepted_by_the_storage_coercion(self):
        """The value must survive `_safe_datetime` unchanged (still UTC)."""
        from storage.backend import _safe_datetime
        value = utc_now()
        coerced = _safe_datetime(value)
        self.assertEqual(coerced, value)
        self.assertEqual(coerced.utcoffset(), timedelta(0))


if __name__ == '__main__':
    unittest.main()
