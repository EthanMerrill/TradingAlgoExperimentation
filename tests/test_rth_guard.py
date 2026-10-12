#!/usr/bin/env python3
"""Tests for RTH clock correctness and the entry-time RTH guard (Step 1).

Covers the regression that let after-hours entries through: naive datetimes
were treated as UTC, shifting the RTH window by the host's UTC offset.
"""
import os
import sys
import unittest
from datetime import datetime
from unittest.mock import Mock, patch

import pytz

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from utils.calendar import (  # noqa: E402
    is_regular_trading_hours,
    is_session_ended,
    to_eastern,
)
from trading_engine import TradingEngine, TradingOpportunity  # noqa: E402

ET = pytz.timezone("US/Eastern")


def _et(*args) -> datetime:
    return ET.localize(datetime(*args))


class TestRthClock(unittest.TestCase):
    def test_naive_interpreted_as_local_not_utc(self):
        # A naive local wall-clock time must round-trip to the same ET time.
        # (Previously 19:21 local was read as 19:21 UTC -> 15:21 ET, wrongly
        # inside RTH.)
        naive = datetime(2026, 10, 9, 19, 21)
        self.assertEqual(
            to_eastern(naive).strftime("%H:%M"), "19:21")

    def test_aware_utc_converted(self):
        aware = datetime(2026, 10, 9, 23, 21, tzinfo=pytz.UTC)
        self.assertEqual(to_eastern(aware).strftime("%H:%M"), "19:21")

    def test_is_rth_boundaries(self):
        self.assertTrue(is_regular_trading_hours(_et(2026, 10, 9, 9, 30)))
        self.assertTrue(is_regular_trading_hours(_et(2026, 10, 9, 12, 0)))
        self.assertTrue(is_regular_trading_hours(_et(2026, 10, 9, 16, 0)))
        self.assertFalse(is_regular_trading_hours(_et(2026, 10, 9, 16, 1)))
        # The exact incident time — must NOT be RTH.
        self.assertFalse(is_regular_trading_hours(_et(2026, 10, 9, 19, 21)))

    def test_is_rth_weekend(self):
        self.assertFalse(is_regular_trading_hours(_et(2026, 10, 10, 11, 0)))

    def test_session_ended_boundaries(self):
        self.assertFalse(is_session_ended(_et(2026, 10, 9, 10, 0)))
        self.assertFalse(is_session_ended(_et(2026, 10, 9, 16, 0)))
        self.assertTrue(is_session_ended(_et(2026, 10, 9, 16, 1)))
        self.assertFalse(is_session_ended(_et(2026, 10, 10, 18, 0)))


def _opportunity() -> TradingOpportunity:
    return TradingOpportunity(
        symbol="AAPL",
        current_rsi=25.0,
        target_rsi_lower=30,
        target_rsi_upper=70,
        rsi_period=14,
        backtest_return=0.15,
        alpha=0.05,
        win_rate=0.9,
        entry_price=100.0,
        stop_loss_price=95.0,
        take_profit_price=110.0,
        num_trades=10,
    )


class TestEntryRthGuard(unittest.TestCase):
    def setUp(self):
        with patch('trading_engine.data_provider'):
            self.engine = TradingEngine()
        self.engine._positions_manager = Mock()
        self.engine._positions_manager.positions = []
        self.engine._positions_manager.open_position = Mock()
        self.engine.ENTRY_FILL_WAIT_SECONDS = 0
        self.engine.dry_run = False
        self.engine.trading_client = Mock()
        self.engine.trading_client.submit_order.return_value = Mock(id="o1")

    def test_entry_blocked_outside_rth(self):
        self.assertTrue(self.engine.enforce_rth_entries)
        with patch('trading_engine.is_regular_trading_hours',
                   return_value=False), \
                patch('trading_engine.storage.save_orders') as save:
            result = self.engine.place_buy_order(_opportunity(), 5)
        self.assertFalse(result)
        self.engine.trading_client.submit_order.assert_not_called()
        self.engine._positions_manager.open_position.assert_not_called()
        save.assert_not_called()

    def test_entry_allowed_inside_rth(self):
        with patch('trading_engine.is_regular_trading_hours',
                   return_value=True), \
                patch.object(self.engine, '_make_unique_client_order_id',
                             return_value="AAPL-BUY-TEST"), \
                patch('trading_engine.storage.save_orders'):
            result = self.engine.place_buy_order(_opportunity(), 5)
        self.assertTrue(result)
        self.engine.trading_client.submit_order.assert_called_once()

    def test_guard_disabled_allows_entry_outside_rth(self):
        self.engine.enforce_rth_entries = False
        with patch('trading_engine.is_regular_trading_hours',
                   return_value=False), \
                patch.object(self.engine, '_make_unique_client_order_id',
                             return_value="AAPL-BUY-TEST"), \
                patch('trading_engine.storage.save_orders'):
            result = self.engine.place_buy_order(_opportunity(), 5)
        self.assertTrue(result)
        self.engine.trading_client.submit_order.assert_called_once()


if __name__ == '__main__':
    unittest.main()
