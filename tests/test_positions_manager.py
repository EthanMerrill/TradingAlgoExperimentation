#!/usr/bin/env python3
"""Unit tests for positions module (current API)."""
import os
import sys
import unittest
from datetime import datetime
from unittest.mock import Mock

import pandas as pd

# Add app path before imports.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from positions import Position, PositionsManager  # noqa: E402


class TestPosition(unittest.TestCase):
    def test_position_creation(self):
        position = Position(
            symbol="AAPL",
            quantity=100.0,
            entry_price=150.0,
            current_price=155.0,
            current_rsi=45.0,
            entry_date=datetime(2025, 6, 14),
            alpha=0.1,
            rsi_period=14,
            rsi_lower=30,
            rsi_upper=70,
            stop_loss_price=140.0,
            take_profit_price=165.0,
        )

        self.assertEqual(position.symbol, "AAPL")
        self.assertEqual(position.quantity, 100.0)
        self.assertEqual(position.rsi_period, 14)

    def test_position_optional_fields(self):
        position = Position(
            symbol="TSLA",
            quantity=50.0,
            entry_price=800.0,
            current_price=850.0,
            current_rsi=40.0,
            entry_date=datetime(2025, 6, 14),
            alpha=0.2,
            rsi_period=14,
            rsi_lower=30,
            rsi_upper=70,
        )

        self.assertIsNone(position.stop_loss_price)
        self.assertIsNone(position.take_profit_price)
        self.assertIsNone(position.exit_reason)


class TestPositionsManager(unittest.TestCase):
    def setUp(self):
        self.cloud = Mock()
        self.data = Mock()
        self.manager = PositionsManager(self.cloud, self.data)

    def _empty_positions_df(self):
        return pd.DataFrame(columns=['symbol', 'qty', 'avg_entry_price', 'current_price', 'market_value'])

    def _empty_cloud_df(self):
        return pd.DataFrame(columns=['symbol'])

    def test_open_position_and_duplicate_guard(self):
        p = Position(
            symbol="AAPL",
            quantity=10.0,
            entry_price=100.0,
            current_price=101.0,
            current_rsi=45.0,
            entry_date=datetime.now(),
            alpha=0.1,
            rsi_period=14,
            rsi_lower=30,
            rsi_upper=70,
            closed=False,
        )

        self.manager.open_position(p)
        self.manager.open_position(p)

        self.assertEqual(len(self.manager.positions), 1)

    def test_close_position_removes_from_in_memory_list(self):
        p = Position(
            symbol="AAPL",
            quantity=10.0,
            entry_price=100.0,
            current_price=101.0,
            current_rsi=45.0,
            entry_date=datetime.now(),
            alpha=0.1,
            rsi_period=14,
            rsi_lower=30,
            rsi_upper=70,
            closed=False,
        )
        self.manager.positions = [p]
        self.cloud.get_latest_positions_df.return_value = pd.DataFrame({
            'symbol': ['AAPL'],
            'entry_price': [100.0],
            'current_price': [101.0],
            'closed': [False],
        })

        self.manager.close_position("AAPL")

        self.assertEqual(len(self.manager.positions), 0)

    def test_get_and_reconcile_positions_initializes_from_alpaca_when_cloud_empty(self):
        self.data.get_current_positions_df.return_value = pd.DataFrame({
            'symbol': ['AAPL'],
            'qty': [10.0],
            'avg_entry_price': [100.0],
            'current_price': [101.0],
            'market_value': [1010.0],
        })

        def cloud_side_effect(is_open):
            return self._empty_cloud_df() if is_open else self._empty_cloud_df()

        self.cloud.get_latest_positions_df.side_effect = cloud_side_effect

        open_positions = self.manager.get_and_reconcile_positions()

        self.assertEqual(len(open_positions), 1)
        self.assertEqual(open_positions[0].symbol, 'AAPL')
        self.assertFalse(open_positions[0].closed)

    def test_get_and_reconcile_positions_marks_cloud_only_positions_closed(self):
        self.data.get_current_positions_df.return_value = self._empty_positions_df()

        open_cloud_df = pd.DataFrame({
            'symbol': ['AAPL'],
            'shares': [10.0],
            'entry_price': [100.0],
            'current_price': [90.0],
            'position_value': [900.0],
            'current_rsi': [40.0],
            'entry_date': [datetime.now()],
            'rsi_period': [14],
            'rsi_lower': [30],
            'rsi_upper': [70],
            'alpha': [0.1],
            'stop_loss_price': [95.0],
            'take_profit_price': [110.0],
            'exit_date': [None],
            'exit_price': [pd.NA],
            'realized_return': [pd.NA],
            'closed': [False],
        })

        def cloud_side_effect(is_open):
            if is_open:
                return open_cloud_df.copy()
            return self._empty_cloud_df()

        self.cloud.get_latest_positions_df.side_effect = cloud_side_effect

        open_positions = self.manager.get_and_reconcile_positions()

        self.assertEqual(len(open_positions), 0)
        self.assertTrue(any(p.closed for p in self.manager.positions))
        closed = [p for p in self.manager.positions if p.closed]
        self.assertTrue(any(p.exit_reason == 'broker_closed' for p in closed))


if __name__ == '__main__':
    unittest.main()
