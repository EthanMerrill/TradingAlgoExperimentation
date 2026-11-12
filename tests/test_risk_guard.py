"""Tests for the portfolio kill switch (app/risk.py) and late-fill handling."""
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from risk import RiskGuard  # noqa: E402
from trading_engine import TradingEngine, TradingOpportunity  # noqa: E402


def _cfg(**kw):
    base = dict(MAX_DAILY_LOSS_PCT=0.03, MAX_DRAWDOWN_HALT_PCT=0.15,
                TRADING_HALTED=False)
    base.update(kw)
    return SimpleNamespace(**base)


def _guard(info, meta=None, cfg=None):
    return RiskGuard(cfg or _cfg(), lambda: info,
                     (lambda **kw: meta) if meta is not None else None)


class TestRiskGuard(unittest.TestCase):
    def setUp(self):
        os.environ.pop('TRADING_HALTED', None)

    def tearDown(self):
        os.environ.pop('TRADING_HALTED', None)

    def test_ok(self):
        self.assertIsNone(_guard(
            {'equity': 99000.0, 'last_equity': 100000.0}).entry_block_reason())

    def test_daily_loss_blocks(self):
        r = _guard({'equity': 96000.0, 'last_equity': 100000.0}
                   ).entry_block_reason()
        self.assertIn('daily loss', r)

    def test_missing_last_equity_does_not_block(self):
        self.assertIsNone(_guard({'equity': 1.0}).entry_block_reason())

    def test_empty_account_info_does_not_block(self):
        self.assertIsNone(_guard({}).entry_block_reason())

    def test_drawdown_blocks(self):
        meta = [{'metadata': {'portfolio_value': 100000.0}},
                {'metadata': {'portfolio_value': 90000.0}}]
        r = _guard({'equity': 80000.0, 'last_equity': 80000.0}, meta
                   ).entry_block_reason()
        self.assertIn('drawdown', r)

    def test_drawdown_within_limit_ok(self):
        meta = [{'metadata': {'portfolio_value': 100000.0}}]
        self.assertIsNone(_guard(
            {'equity': 95000.0, 'last_equity': 95000.0}, meta
        ).entry_block_reason())

    def test_no_history_skips_drawdown(self):
        self.assertIsNone(_guard(
            {'equity': 1000.0, 'last_equity': 1000.0}, []
        ).entry_block_reason())

    def test_manual_halt_env_read_at_check_time(self):
        g = _guard({'equity': 1.0})
        self.assertIsNone(g.entry_block_reason())
        os.environ['TRADING_HALTED'] = '1'
        self.assertIn('manual halt', g.entry_block_reason())

    def test_manual_halt_config(self):
        g = _guard({'equity': 1.0}, cfg=_cfg(TRADING_HALTED=True))
        self.assertIn('manual halt', g.entry_block_reason())

    def test_account_blocked(self):
        for flag in ('trading_blocked', 'account_blocked',
                     'trade_suspended_by_user'):
            r = _guard({'equity': 1.0, flag: True}).entry_block_reason()
            self.assertIn(flag, r)

    def test_result_cached(self):
        calls = []

        def info():
            calls.append(1)
            return {'equity': 1.0}
        g = RiskGuard(_cfg(), info)
        g.entry_block_reason()
        g.entry_block_reason()
        self.assertEqual(len(calls), 1)


def _opp():
    return TradingOpportunity(
        symbol="AAPL", current_rsi=25.0, target_rsi_lower=30,
        target_rsi_upper=70, rsi_period=14, backtest_return=0.15,
        alpha=0.05, win_rate=0.9, entry_price=100.0,
        stop_loss_price=95.0, take_profit_price=110.0, num_trades=10)


class _EngineFixture(unittest.TestCase):
    def setUp(self):
        with patch('trading_engine.data_provider'):
            self.engine = TradingEngine()
        self.engine._positions_manager = Mock()
        self.engine.enforce_rth_entries = False
        self.engine.set_dry_run_mode(False)
        self.engine.trading_client = Mock()
        self.engine.trading_client.submit_order.return_value = \
            SimpleNamespace(id="order_1")


class TestEngineRiskWiring(_EngineFixture):
    def test_blocked_entry_not_submitted(self):
        self.engine._risk_guard = Mock()
        self.engine._risk_guard.entry_block_reason.return_value = "daily loss"
        self.assertFalse(self.engine.place_buy_order(_opp(), 5))
        self.engine.trading_client.submit_order.assert_not_called()
        self.assertIn("daily loss", self.engine._last_order_error)

    def test_exit_not_gated_by_risk(self):
        self.engine._risk_guard = Mock()
        self.engine._risk_guard.entry_block_reason.return_value = "halt"
        pos = SimpleNamespace(symbol="AAPL", quantity=5.0, side="long")
        try:
            self.engine.close_position_at_market(pos)
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        self.engine._risk_guard.entry_block_reason.assert_not_called()


class TestLateFillAfterCancel(_EngineFixture):
    def _run(self, wait_result, orders):
        self.engine.trading_client.get_order_by_id.side_effect = orders
        with patch.object(self.engine, '_make_unique_client_order_id',
                          return_value="AAPL-BUY-TEST"), \
                patch.object(self.engine, '_wait_for_entry_fill',
                             return_value=(None, None, 'new')), \
                patch.object(self.engine, '_cancel_entry_order'), \
                patch.object(self.engine, '_wait_for_orders_cancelled',
                             return_value=wait_result) as waiter, \
                patch.object(self.engine, 'place_oco_close_order',
                             return_value=True) as oco, \
                patch('trading_engine.storage.save_orders'):
            result = self.engine.place_buy_order(_opp(), 5)
        return result, waiter, oco

    def test_pending_cancel_then_filled_is_recorded(self):
        filled = SimpleNamespace(status='filled', filled_qty='5',
                                 filled_avg_price='100.1')
        result, waiter, oco = self._run(True, [filled])
        self.assertTrue(result)
        waiter.assert_called_once()
        created = self.engine._positions_manager.open_position.call_args[0][0]
        self.assertEqual(created.quantity, 5.0)
        oco.assert_called_once_with("AAPL", 5.0, 95.0, 110.0, side="long")

    def test_not_terminal_after_wait_records_nothing(self):
        pending = SimpleNamespace(status='pending_cancel', filled_qty='0',
                                  filled_avg_price=None)
        with self.assertLogs('trading_engine', level='ERROR') as cm:
            result, _, oco = self._run(False, [pending])
        self.assertFalse(result)
        self.assertTrue(any('reconcile' in m for m in cm.output))
        oco.assert_not_called()
        self.engine._positions_manager.open_position.assert_not_called()

    def test_real_wait_sequence_pending_cancel_then_filled(self):
        seq = [SimpleNamespace(status='pending_cancel', filled_qty='0',
                               filled_avg_price=None),
               SimpleNamespace(status='filled', filled_qty='5',
                               filled_avg_price='100.1'),
               SimpleNamespace(status='filled', filled_qty='5',
                               filled_avg_price='100.1')]
        self.engine.trading_client.get_order_by_id.side_effect = seq
        with patch.object(self.engine, '_make_unique_client_order_id',
                          return_value="AAPL-BUY-TEST"), \
                patch.object(self.engine, '_wait_for_entry_fill',
                             return_value=(None, None, 'new')), \
                patch.object(self.engine, '_cancel_entry_order'), \
                patch('trading_engine.time.sleep'), \
                patch.object(self.engine, 'place_oco_close_order',
                             return_value=True), \
                patch('trading_engine.storage.save_orders'):
            self.assertTrue(self.engine.place_buy_order(_opp(), 5))


if __name__ == '__main__':
    unittest.main()
