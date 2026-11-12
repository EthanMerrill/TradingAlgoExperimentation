#!/usr/bin/env python3
"""Regression tests for position reconciliation return values / safety."""

import os
import sys
import unittest
from unittest.mock import Mock

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from positions import PositionsFetchError, PositionsManager  # noqa: E402


def _manager(broker_df=None, **provider_kw):
    storage = Mock()
    storage.load_position_metadata.return_value = []
    storage.load_closed_trades.return_value = []
    data = Mock()
    data.get_current_positions_df.return_value = (
        broker_df if broker_df is not None else pd.DataFrame())
    data.get_active_orders.return_value = pd.DataFrame()
    for k, v in provider_kw.items():
        setattr(data, k, v)
    return PositionsManager(storage, data), storage, data


class TestPositionsReconcileRegression(unittest.TestCase):
    def test_reconcile_returns_empty_list_when_no_positions(self):
        manager, _, _ = _manager()
        result = manager.get_and_reconcile_positions()
        self.assertIsInstance(result, list)
        self.assertEqual(result, [])

    def test_empty_broker_with_unreadable_stores_does_not_write(self):
        manager, storage, _ = _manager()
        storage.load_position_metadata.return_value = None
        storage.load_closed_trades.return_value = None
        self.assertEqual(manager.get_and_reconcile_positions(), [])
        storage.save_closed_trade.assert_not_called()
        storage.delete_position_metadata.assert_not_called()

    def test_provider_error_is_not_an_empty_book(self):
        """The provider swallowing an API error into an empty frame is the
        failure mode that used to close every position; reconcile must ask for
        errors to propagate and refuse to act on them."""
        manager, storage, data = _manager()
        storage.load_position_metadata.return_value = [
            {"symbol": "AAPL", "side": "long", "quantity": 1.0,
             "entry_price": 10.0}]
        data.get_current_positions_df.side_effect = RuntimeError("boom")
        with self.assertRaises(PositionsFetchError):
            manager.get_and_reconcile_positions()
        data.get_current_positions_df.assert_called_with(raise_on_error=True)
        storage.save_closed_trade.assert_not_called()
        storage.delete_position_metadata.assert_not_called()
        storage.save_position_metadata.assert_not_called()


if __name__ == '__main__':
    unittest.main()
