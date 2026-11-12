#!/usr/bin/env python3
"""Tests for the unsettled-entry watch list (cancel never settled)."""
import os
import sys
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from trading_engine import TradingEngine, TradingOpportunity  # noqa: E402
from positions import Position  # noqa: E402


def _opp(symbol="AAPL"):
    return TradingOpportunity(
        symbol=symbol, current_rsi=25.0, target_rsi_lower=30,
        target_rsi_upper=70, rsi_period=14, backtest_return=0.15,
        alpha=0.05, win_rate=0.9, entry_price=100.0,
        stop_loss_price=95.0, take_profit_price=110.0, num_trades=10,
        strategy_name="my_strategy")


def _order(status, qty="0", price=None, filled_at=None):
    return SimpleNamespace(status=status, filled_qty=qty,
                           filled_avg_price=price, filled_at=filled_at)


class TestUnsettledEntries(unittest.TestCase):
    def setUp(self):
        with patch('trading_engine.data_provider'):
            self.engine = TradingEngine()
        self.engine._positions_manager = Mock()
        self.engine._positions_manager.positions = []
        self.engine.ENTRY_FILL_WAIT_SECONDS = 0
        self.engine.dry_run = False
        self.engine.enforce_rth_entries = False
        self.engine._risk_guard = Mock(entry_block_reason=Mock(return_value=None))
        self.engine.trading_client = Mock()
        self.engine.trading_client.submit_order.return_value = SimpleNamespace(
            id="order_1")
        self.engine._unsettled_loaded = True
        patcher = patch('trading_engine.storage')
        self.storage = patcher.start()
        self.addCleanup(patcher.stop)

    def _place_unsettled(self, settle_cfg=None):
        """Place a buy whose cancel never settles."""
        with patch.object(self.engine, '_make_unique_client_order_id',
                          return_value="AAPL-BUY-TEST"), \
                patch.object(self.engine, '_wait_for_entry_fill',
                             return_value=(None, None, 'new')), \
                patch.object(self.engine, '_cancel_entry_order'), \
                patch.object(self.engine, '_wait_for_orders_cancelled',
                             return_value=False) as wait, \
                patch.object(self.engine, '_requery_entry_fill',
                             return_value=(None, None)), \
                patch.object(self.engine, 'place_oco_close_order') as oco:
            result = self.engine.place_buy_order(_opp(), 5)
        return result, wait, oco

    def test_settle_timeout_registers_watch_no_position(self):
        result, _, oco = self._place_unsettled()
        self.assertFalse(result)
        self.engine._positions_manager.open_position.assert_not_called()
        oco.assert_not_called()
        rec = self.engine._unsettled_entries["AAPL-BUY-TEST"]
        self.assertEqual(rec['order_id'], "order_1")
        self.assertEqual(rec['shares'], 5)
        self.assertEqual(rec['quantity_sign'], 1)
        self.assertEqual(rec['opportunity']['strategy_name'], "my_strategy")
        self.assertEqual(rec['opportunity']['stop_loss_price'], 95.0)
        self.storage.save_unsettled_entries.assert_called_once()

    def test_next_tick_filled_records_position_and_arms_oco(self):
        self._place_unsettled()
        filled_at = datetime(2026, 11, 12, 15, 0, tzinfo=timezone.utc)
        self.engine.trading_client.get_order_by_id.return_value = _order(
            'filled', '5', '100.5', filled_at)
        with patch.object(self.engine, 'place_oco_close_order',
                          return_value=True) as oco:
            summary = self.engine.resolve_unsettled_entries()
        self.assertEqual(summary['filled'], 1)
        pos = self.engine._positions_manager.open_position.call_args[0][0]
        self.assertEqual(pos.quantity, 5.0)
        self.assertEqual(pos.entry_price, 100.5)
        self.assertEqual(pos.entry_date, filled_at)
        self.assertEqual(pos.strategy_name, "my_strategy")
        oco.assert_called_once_with("AAPL", 5.0, 95.0, 110.0, side="long")
        self.assertEqual(self.engine._unsettled_entries, {})
        self.storage.delete_unsettled_entries.assert_called_once_with(
            ["AAPL-BUY-TEST"])

    def test_zero_filled_is_removed(self):
        self._place_unsettled()
        self.engine.trading_client.get_order_by_id.return_value = _order(
            'canceled', '0')
        with patch.object(self.engine, 'place_oco_close_order') as oco:
            self.engine.resolve_unsettled_entries()
        oco.assert_not_called()
        self.engine._positions_manager.open_position.assert_not_called()
        self.assertEqual(self.engine._unsettled_entries, {})

    def test_still_pending_kept_and_old_ones_recancelled(self):
        self._place_unsettled()
        self.engine.trading_client.get_order_by_id.return_value = _order(
            'pending_cancel')
        with patch.object(self.engine, '_cancel_entry_order') as cancel:
            self.engine.resolve_unsettled_entries()
            cancel.assert_not_called()
            self.engine._unsettled_entries["AAPL-BUY-TEST"]['first_seen'] = (
                datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
            summary = self.engine.resolve_unsettled_entries()
            cancel.assert_called_once_with("order_1")
        self.assertEqual(summary['pending'], 1)
        self.assertIn("AAPL-BUY-TEST", self.engine._unsettled_entries)

    def test_already_adopted_position_not_duplicated(self):
        self._place_unsettled()
        adopted = Position(
            symbol="AAPL", quantity=5.0, entry_price=100.5,
            current_price=101.0, current_rsi=0.0,
            entry_date=datetime.now(timezone.utc), alpha=0.0, rsi_period=14,
            rsi_lower=30, rsi_upper=70)
        self.engine._positions_manager.positions = [adopted]
        self.engine.trading_client.get_order_by_id.return_value = _order(
            'filled', '5', '100.5')
        with patch.object(self.engine, '_get_active_orders_for_symbol',
                          return_value=[]), \
                patch.object(self.engine, 'place_oco_close_order',
                             return_value=True) as oco:
            self.engine.resolve_unsettled_entries()
        self.engine._positions_manager.open_position.assert_not_called()
        self.assertEqual(adopted.strategy_name, "my_strategy")
        self.assertEqual(adopted.stop_loss_price, 95.0)
        self.assertEqual(adopted.take_profit_price, 110.0)
        self.assertEqual(adopted.client_order_id, "AAPL-BUY-TEST")
        oco.assert_called_once()

    def test_adopted_with_existing_protection_skips_oco(self):
        self._place_unsettled()
        adopted = Position(
            symbol="AAPL", quantity=5.0, entry_price=100.5,
            current_price=101.0, current_rsi=0.0,
            entry_date=datetime.now(timezone.utc), alpha=0.0, rsi_period=14,
            rsi_lower=30, rsi_upper=70)
        self.engine._positions_manager.positions = [adopted]
        self.engine.trading_client.get_order_by_id.return_value = _order(
            'filled', '5', '100.5')
        with patch.object(self.engine, '_get_active_orders_for_symbol',
                          return_value=[Mock()]), \
                patch.object(self.engine, 'place_oco_close_order') as oco:
            self.engine.resolve_unsettled_entries()
        oco.assert_not_called()

    def test_dry_run_skips(self):
        self._place_unsettled()
        self.engine.dry_run = True
        summary = self.engine.resolve_unsettled_entries()
        self.assertEqual(summary, {'resolved': 0, 'filled': 0, 'pending': 0})
        self.engine.trading_client.get_order_by_id.assert_not_called()
        self.assertIn("AAPL-BUY-TEST", self.engine._unsettled_entries)

    def test_watch_list_loaded_from_storage_after_restart(self):
        self._place_unsettled()
        persisted = dict(self.engine._unsettled_entries["AAPL-BUY-TEST"])
        self.engine._unsettled_entries.clear()
        self.engine._unsettled_loaded = False
        self.storage.load_unsettled_entries.return_value = [persisted]
        self.engine.trading_client.get_order_by_id.return_value = _order(
            'filled', '5', '100.5')
        with patch.object(self.engine, 'place_oco_close_order',
                          return_value=True):
            summary = self.engine.resolve_unsettled_entries()
        self.assertEqual(summary['filled'], 1)

    def test_config_settle_seconds_respected(self):
        with patch('trading_engine.globalConfig') as cfg:
            cfg.ENTRY_CANCEL_SETTLE_SECONDS = 42
            _, wait, _ = self._place_unsettled()
        self.assertEqual(wait.call_args.kwargs['timeout'], 42.0)

    def test_session_start_resolves(self):
        with patch.object(self.engine, 'resolve_unsettled_entries') as res:
            self.engine._positions_manager.get_and_reconcile_positions.return_value = []
            self.engine.execute_trading_session([])
        res.assert_called_once()


if __name__ == '__main__':
    unittest.main()
