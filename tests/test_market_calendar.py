#!/usr/bin/env python3
"""NYSE market-calendar tests (built-in rules + mocked Alpaca calendar)."""
import os
import sys
import unittest
from datetime import date, datetime, time as dtime
from types import SimpleNamespace
from unittest.mock import Mock

import pytz

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from utils import datetime_ as dt_mod  # noqa: E402
from utils.calendar import is_regular_trading_hours, is_session_ended  # noqa: E402
from utils.datetime_ import (  # noqa: E402
    is_trading_day, nyse_holidays, reset_calendar_cache, set_calendar_client)

ET = pytz.timezone("US/Eastern")


def _et(*a):
    return ET.localize(datetime(*a))


class TestBuiltinRules(unittest.TestCase):
    def setUp(self):
        set_calendar_client(None)  # PYTEST_CURRENT_TEST blocks real client

    def test_columbus_and_veterans_open(self):
        self.assertTrue(is_trading_day(datetime(2026, 10, 12)))
        self.assertTrue(is_trading_day(datetime(2026, 11, 11)))

    def test_good_friday_closed(self):
        self.assertFalse(is_trading_day(datetime(2026, 4, 3)))
        self.assertFalse(is_trading_day(datetime(2025, 4, 18)))

    def test_2026_holidays(self):
        h = nyse_holidays(2026)
        for d in [(1, 1), (1, 19), (2, 16), (4, 3), (5, 25), (6, 19),
                  (7, 3), (9, 7), (11, 26), (12, 25)]:
            self.assertIn(date(2026, *d), h, d)
        self.assertEqual(len(h), 10)

    def test_observed_rules(self):
        self.assertIn(date(2027, 12, 24), nyse_holidays(2027))  # Xmas Sat
        self.assertIn(date(2027, 7, 5), nyse_holidays(2027))    # Jul 4 Sun
        self.assertNotIn(date(2027, 12, 31), nyse_holidays(2027))
        self.assertNotIn(date(2022, 12, 31), nyse_holidays(2022))
        self.assertIn(date(2022, 12, 26), nyse_holidays(2022))
        self.assertNotIn(date(2021, 6, 18), nyse_holidays(2021))  # pre-2022

    def test_early_closes(self):
        self.assertFalse(is_regular_trading_hours(_et(2026, 11, 27, 13, 1)))
        self.assertTrue(is_regular_trading_hours(_et(2026, 11, 27, 12, 59)))
        self.assertTrue(is_session_ended(_et(2026, 11, 27, 13, 1)))
        self.assertFalse(is_session_ended(_et(2026, 11, 27, 12, 0)))
        self.assertFalse(is_regular_trading_hours(_et(2026, 12, 24, 14, 0)))
        self.assertFalse(is_regular_trading_hours(_et(2025, 7, 3, 14, 0)))
        # Normal day stays open until 16:00.
        self.assertTrue(is_regular_trading_hours(_et(2026, 11, 12, 15, 30)))

    def test_holiday_not_rth(self):
        self.assertFalse(is_regular_trading_hours(_et(2026, 12, 25, 11, 0)))
        self.assertFalse(is_session_ended(_et(2026, 12, 25, 17, 0)))


class TestAlpacaCalendar(unittest.TestCase):
    def tearDown(self):
        set_calendar_client(None)

    def _client(self):
        Calendar = SimpleNamespace  # same attrs as alpaca Calendar model
        client = Mock()
        client.get_calendar.return_value = [
            Calendar(date=date(2026, 11, 12), open=datetime(2026, 11, 12, 9, 30),
                     close=datetime(2026, 11, 12, 15, 0)),
            Calendar(date=date(2026, 11, 13), open=datetime(2026, 11, 13, 9, 30),
                     close=datetime(2026, 11, 13, 16, 0)),
        ]
        return client

    def test_api_sessions_used_and_cached(self):
        client = self._client()
        set_calendar_client(client)
        self.assertFalse(is_regular_trading_hours(_et(2026, 11, 12, 15, 30)))
        self.assertTrue(is_regular_trading_hours(_et(2026, 11, 13, 15, 30)))
        self.assertTrue(is_session_ended(_et(2026, 11, 12, 15, 1)))
        # Date absent from API response => closed (even a weekday).
        self.assertFalse(is_trading_day(datetime(2026, 11, 16)))
        self.assertEqual(client.get_calendar.call_count, 1)

    def test_api_error_falls_back_to_rules(self):
        client = Mock()
        client.get_calendar.side_effect = RuntimeError("network down")
        set_calendar_client(client)
        self.assertTrue(is_trading_day(datetime(2026, 10, 12)))
        self.assertFalse(is_trading_day(datetime(2026, 4, 3)))
        self.assertEqual(client.get_calendar.call_count, 1)  # no retry storm


if __name__ == "__main__":
    unittest.main()
