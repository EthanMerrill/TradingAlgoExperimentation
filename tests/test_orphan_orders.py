#!/usr/bin/env python3
"""Tests for orphan-order detection, classification, and cancellation (Step 2).

Covers:
  * data_provider.get_active_orders (all types, non-terminal only)
  * data_provider.get_open_orders(tag_orphans=True) tagging
  * PositionsManager._cancel_orphan_orders (authority + config gate)
"""
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from data_provider import DataProvider  # noqa: E402
from positions import PositionsManager  # noqa: E402


def _dp_with_client(tc) -> DataProvider:
    """Build a DataProvider whose trading client is ``tc`` (no network/init)."""
    dp = DataProvider.__new__(DataProvider)
    dp._historical_client = None
    dp._trading_client = tc
    dp._clients_initialized = True
    dp._rate_limit_delay = 0.0
    dp._shortable_cache = {}
    return dp


def _order(symbol, oid, otype, side="sell", status="new",
           limit=10.0, stop=None, order_class="simple"):
    return SimpleNamespace(
        symbol=symbol, id=oid, type=otype, side=side, status=status,
        qty=1, limit_price=limit, stop_price=stop,
        order_class=order_class, created_at=None)


class TestGetActiveOrders(unittest.TestCase):
    def test_excludes_terminal_and_includes_market(self):
        tc = Mock()
        tc.get_orders.return_value = [
            _order("MSFT", "m1", "market", side="buy"),
            _order("AAPL", "a1", "limit", status="filled"),
            _order("AAPL", "a2", "limit", status="new"),
        ]
        df = _dp_with_client(tc).get_active_orders()
        self.assertIn("m1", list(df["order_id"]))
        self.assertIn("a2", list(df["order_id"]))
        self.assertNotIn("a1", list(df["order_id"]))

    def test_empty_on_client_error(self):
        tc = Mock()
        tc.get_orders.side_effect = RuntimeError("boom")
        self.assertTrue(_dp_with_client(tc).get_active_orders().empty)


class TestTagOrphanLegs(unittest.TestCase):
    def test_tags_symbol_without_position(self):
        tc = Mock()
        tc.get_orders.return_value = [
            _order("AAPL", "a1", "limit"),
            _order("MSFT", "m1", "limit"),
        ]
        dp = _dp_with_client(tc)
        dp.get_current_positions_df = Mock(
            return_value=pd.DataFrame({"symbol": ["AAPL"]}))

        df = dp.get_open_orders(tag_orphans=True)
        by_symbol = {r["symbol"]: r["leg_type"] for _, r in df.iterrows()}
        self.assertEqual(by_symbol["AAPL"], "take_profit")
        self.assertEqual(by_symbol["MSFT"], "orphan")

    def test_no_tagging_by_default(self):
        tc = Mock()
        tc.get_orders.return_value = [_order("MSFT", "m1", "limit")]
        dp = _dp_with_client(tc)
        dp.get_current_positions_df = Mock(
            return_value=pd.DataFrame({"symbol": ["AAPL"]}))
        df = dp.get_open_orders()
        self.assertEqual(df.iloc[0]["leg_type"], "take_profit")


class TestCancelOrphanOrders(unittest.TestCase):
    def setUp(self):
        self.cloud = Mock()
        self.data = Mock()
        self.manager = PositionsManager(self.cloud, self.data)

    def test_cancels_only_symbols_without_position(self):
        self.data.get_active_orders.return_value = pd.DataFrame([
            {"symbol": "AAPL", "order_id": "a1",
             "order_type": "limit", "side": "sell"},
            {"symbol": "MSFT", "order_id": "m1",
             "order_type": "limit", "side": "sell"},
        ])
        self.manager._cancel_orphan_orders({"AAPL"})
        self.data.cancel_order.assert_called_once_with("m1")

    def test_respects_config_flag(self):
        self.data.get_active_orders.return_value = pd.DataFrame([
            {"symbol": "MSFT", "order_id": "m1",
             "order_type": "limit", "side": "sell"},
        ])
        with patch("positions.globalConfig") as gc:
            gc.CANCEL_ORPHAN_ORDERS = False
            self.manager._cancel_orphan_orders(set())
        self.data.cancel_order.assert_not_called()

    def test_noop_when_provider_lacks_methods(self):
        manager = PositionsManager(Mock(), Mock(spec=[]))
        manager._cancel_orphan_orders(set())  # must not raise

    def test_noop_on_non_dataframe_result(self):
        self.data.get_active_orders.return_value = None
        self.manager._cancel_orphan_orders(set())
        self.data.cancel_order.assert_not_called()


if __name__ == "__main__":
    unittest.main()
