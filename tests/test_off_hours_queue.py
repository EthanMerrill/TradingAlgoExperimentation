#!/usr/bin/env python3
"""Tests for the off-hours entry queue.

When RTH enforcement blocks entries outside regular trading hours, entry
signals are queued in ``pending_signals`` and submitted at the next market
open instead of being dropped (daily cycles run at 01:30 ET, which can never
submit entries under RTH enforcement).
"""
import os
import sys
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

import pytz

from trading_engine import TradingEngine, TradingOpportunity  # noqa: E402

ET = pytz.timezone("US/Eastern")


def _opportunity(symbol="AAPL", entry_price=100.0) -> TradingOpportunity:
    return TradingOpportunity(
        symbol=symbol,
        current_rsi=25.0,
        target_rsi_lower=30,
        target_rsi_upper=70,
        rsi_period=14,
        backtest_return=0.15,
        alpha=0.05,
        win_rate=0.9,
        entry_price=entry_price,
        stop_loss_price=entry_price * 0.95,
        take_profit_price=entry_price * 1.10,
        num_trades=10,
    )


class _EngineFixture(unittest.TestCase):
    """Common engine setup with storage and sizing mocked."""

    def setUp(self):
        with patch('trading_engine.data_provider'):
            self.engine = TradingEngine()
        self.engine._positions_manager = Mock()
        self.engine._positions_manager.positions = []
        self.engine.ENTRY_FILL_WAIT_SECONDS = 0
        self.engine.dry_run = False
        self.engine.summary = {
            'timestamp': 'x', 'opportunities_found': 1, 'new_positions': 0,
            'orders_placed': 0, 'positions_exited': 0, 'errors': [],
            'dry_run': False, 'orders': [],
        }

        self._size_patcher = patch.object(
            self.engine, 'calculate_position_sizes',
            return_value=[(_opportunity(), 5)])
        self._size_patcher.start()
        self.addCleanup(self._size_patcher.stop)


class TestOffHoursQueueing(_EngineFixture):
    def test_off_hours_entries_are_queued_not_placed(self):
        with patch('trading_engine.is_regular_trading_hours',
                   return_value=False), \
                patch('trading_engine.storage.save_pending_signals',
                      return_value=True) as save:
            result = self.engine._execute_purchases(
                self.engine.summary, [_opportunity()])

        self.assertEqual(result.get('entries_queued'), 1)
        self.assertEqual(result['orders_placed'], 0)
        self.assertEqual(result['new_positions'], 0)
        save.assert_called_once()
        signals = save.call_args[0][0]
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]['symbol'], 'AAPL')
        self.assertEqual(signals[0]['direction'], 'long')
        self.assertEqual(
            signals[0]['opportunity']['entry_price'], 100.0)
        queued_order = next(
            o for o in result['orders'] if o['type'] == 'queued')
        self.assertEqual(queued_order['symbol'], 'AAPL')
        self.assertEqual(queued_order['reason'], 'queued_for_next_open')

    def test_queue_dedupes_within_batch(self):
        with patch('trading_engine.is_regular_trading_hours',
                   return_value=False), \
                patch('trading_engine.storage.save_pending_signals',
                      return_value=True) as save:
            self.engine._execute_purchases(
                self.engine.summary, [_opportunity(), _opportunity()])
        signals = save.call_args[0][0]
        self.assertEqual(len(signals), 1)

    def test_shorts_are_queued_with_direction_short(self):
        with patch('trading_engine.is_regular_trading_hours',
                   return_value=False), \
                patch('trading_engine.storage.save_pending_signals',
                      return_value=True) as save, \
                patch.object(self.engine, 'calculate_short_position_sizes',
                             return_value=[(_opportunity(), 5)]):
            self.engine._execute_shorts(self.engine.summary, [_opportunity()])
        signals = save.call_args[0][0]
        self.assertEqual(signals[0]['direction'], 'short')

    def test_save_failure_falls_back_to_placement(self):
        """If the queue cannot be persisted, keep the old drop/attempt path."""
        with patch('trading_engine.is_regular_trading_hours',
                   return_value=False), \
                patch('trading_engine.storage.save_pending_signals',
                      return_value=False) as save, \
                patch.object(self.engine, 'place_buy_order',
                             return_value=False) as place:
            result = self.engine._execute_purchases(
                self.engine.summary, [_opportunity()])
        save.assert_called_once()
        place.assert_called_once()
        self.assertEqual(result.get('entries_queued', 0), 0)
        # The RTH guard rejects the order — surfaced as an error.
        self.assertTrue(result['errors'])

    def test_dry_run_does_not_queue(self):
        self.engine.dry_run = True
        with patch('trading_engine.is_regular_trading_hours',
                   return_value=False), \
                patch('trading_engine.storage.save_pending_signals') as save, \
                patch.object(self.engine, 'place_buy_order',
                             return_value=True):
            result = self.engine._execute_purchases(
                self.engine.summary, [_opportunity()])
        save.assert_not_called()
        self.assertEqual(result['new_positions'], 1)

    def test_queue_flag_disabled_places_normally(self):
        with patch('trading_engine.is_regular_trading_hours',
                   return_value=False), \
                patch('trading_engine.globalConfig.QUEUE_OFF_HOURS_ENTRIES',
                      False), \
                patch('trading_engine.storage.save_pending_signals') as save, \
                patch.object(self.engine, 'place_buy_order',
                             return_value=True):
            result = self.engine._execute_purchases(
                self.engine.summary, [_opportunity()])
        save.assert_not_called()
        self.assertEqual(result['new_positions'], 1)

    def test_rth_hours_place_normally(self):
        with patch('trading_engine.is_regular_trading_hours',
                   return_value=True), \
                patch('trading_engine.storage.save_pending_signals') as save, \
                patch.object(self.engine, 'place_buy_order',
                             return_value=True):
            result = self.engine._execute_purchases(
                self.engine.summary, [_opportunity()])
        save.assert_not_called()
        self.assertEqual(result['new_positions'], 1)


def _queued_signal(symbol="AAPL", direction="long", days_old=0.0,
                   entry_price=100.0):
    queued_at = datetime.now(timezone.utc) - timedelta(days=days_old)
    return {
        'id': 1,
        'symbol': symbol,
        'direction': direction,
        'strategy_name': 'rsi_mean_reversion',
        'queued_at': queued_at.isoformat(),
        'opportunity': {
            'symbol': symbol,
            'backtest_return': 0.15,
            'alpha': 0.05,
            'win_rate': 0.9,
            'entry_price': entry_price,
            'stop_loss_price': entry_price * 0.95,
            'take_profit_price': entry_price * 1.10,
            'current_rsi': 25.0,
            'target_rsi_lower': 30,
            'target_rsi_upper': 70,
            'rsi_period': 14,
            'num_trades': 10,
            'strategy_name': 'rsi_mean_reversion',
            'direction': direction,
        },
    }


class TestQueueFlush(_EngineFixture):
    def test_flush_requires_rth(self):
        with patch('trading_engine.is_regular_trading_hours',
                   return_value=False), \
                patch('trading_engine.storage.load_pending_signals') as load:
            summary = self.engine.flush_queued_entries()
        load.assert_not_called()
        self.assertEqual(summary, {'flushed': 0, 'submitted': 0,
                                   'expired': 0, 'skipped_stale': 0})

    def test_flush_submits_and_marks_submitted(self):
        placed = []

        def fake_place(opp, shares):
            placed.append(opp.symbol)
            return True

        self.engine.place_buy_order = fake_place
        self._size_patcher.stop()
        with patch('trading_engine.is_regular_trading_hours',
                   return_value=True), \
                patch('trading_engine.storage.load_pending_signals',
                      return_value=[_queued_signal()]), \
                patch('trading_engine.storage.resolve_pending_signals') as res, \
                patch.object(self.engine, 'calculate_position_sizes') as size, \
                patch.object(self.engine, '_get_current_price',
                             return_value=101.0):
            size.return_value = [(_opportunity(), 5)]
            summary = self.engine.flush_queued_entries()
        self.assertEqual(placed, ['AAPL'])
        self.assertEqual(summary['submitted'], 1)
        self.assertEqual(summary['expired'], 0)
        res.assert_called_once_with([1], 'submitted',
                                    reason='flushed_at_open')

    def test_flush_reprices_entry(self):
        """The queued entry price is refreshed from the live quote at open."""
        sized_opps = []

        def fake_size(opportunities):
            sized_opps.extend(opportunities)
            return [(opportunities[0], 5)]

        with patch('trading_engine.is_regular_trading_hours',
                   return_value=True), \
                patch('trading_engine.storage.load_pending_signals',
                      return_value=[_queued_signal(entry_price=100.0)]), \
                patch('trading_engine.storage.resolve_pending_signals'), \
                patch.object(self.engine, 'calculate_position_sizes',
                             side_effect=fake_size), \
                patch.object(self.engine, '_get_current_price',
                             return_value=102.0), \
                patch.object(self.engine, 'place_buy_order', return_value=True):
            self.engine.flush_queued_entries()
        self.assertEqual(sized_opps[0].entry_price, 102.0)
        # SL/TP scale with the refreshed price.
        self.assertAlmostEqual(sized_opps[0].stop_loss_price, 96.9, places=2)
        self.assertAlmostEqual(
            sized_opps[0].take_profit_price, 112.2, places=2)

    def test_flush_expires_stale_signals(self):
        with patch('trading_engine.is_regular_trading_hours',
                   return_value=True), \
                patch('trading_engine.storage.load_pending_signals',
                      return_value=[_queued_signal(days_old=10)]), \
                patch('trading_engine.storage.resolve_pending_signals') as res, \
                patch.object(self.engine, 'place_buy_order') as place, \
                patch.object(self.engine, '_get_current_price',
                             return_value=101.0):
            summary = self.engine.flush_queued_entries()
        place.assert_not_called()
        self.assertEqual(summary['skipped_stale'], 1)
        res.assert_called_once_with([1], 'expired', reason='stale')

    def test_flush_expires_signal_without_live_price(self):
        with patch('trading_engine.is_regular_trading_hours',
                   return_value=True), \
                patch('trading_engine.storage.load_pending_signals',
                      return_value=[_queued_signal()]), \
                patch('trading_engine.storage.resolve_pending_signals') as res, \
                patch.object(self.engine, '_get_current_price',
                             return_value=None):
            summary = self.engine.flush_queued_entries()
        self.assertEqual(summary['expired'], 1)
        res.assert_called_once_with([1], 'expired', reason='no_live_price')

    def test_flush_marks_failed_submissions_expired(self):
        """A submission that fails at open is not retried — it expires."""
        with patch('trading_engine.is_regular_trading_hours',
                   return_value=True), \
                patch('trading_engine.storage.load_pending_signals',
                      return_value=[_queued_signal()]), \
                patch('trading_engine.storage.resolve_pending_signals') as res, \
                patch.object(self.engine, 'calculate_position_sizes',
                             return_value=[(_opportunity(), 5)]), \
                patch.object(self.engine, '_get_current_price',
                             return_value=101.0), \
                patch.object(self.engine, 'place_buy_order',
                             return_value=False):
            summary = self.engine.flush_queued_entries()
        self.assertEqual(summary['submitted'], 0)
        self.assertEqual(summary['expired'], 1)
        res.assert_called_once_with([1], 'expired', reason='flush_failed')

    def test_flush_noop_in_dry_run(self):
        self.engine.dry_run = True
        with patch('trading_engine.is_regular_trading_hours',
                   return_value=True), \
                patch('trading_engine.storage.load_pending_signals') as load:
            summary = self.engine.flush_queued_entries()
        load.assert_not_called()
        self.assertEqual(summary['flushed'], 0)

    def test_exit_orders_do_not_count_as_submitted(self):
        """An opposite-signal exit for the symbol is not an entry submission."""
        def fake_execute(flush_summary, opps):
            self.engine._record_order(
                flush_summary, symbol='AAPL', action='SELL',
                order_type='exit', reason='opposite_signal')
            return flush_summary

        with patch('trading_engine.is_regular_trading_hours',
                   return_value=True), \
                patch('trading_engine.storage.load_pending_signals',
                      return_value=[_queued_signal()]), \
                patch('trading_engine.storage.resolve_pending_signals') as res, \
                patch.object(self.engine, '_execute_purchases',
                             side_effect=fake_execute), \
                patch.object(self.engine, '_get_current_price',
                             return_value=101.0):
            summary = self.engine.flush_queued_entries()
        self.assertEqual(summary['submitted'], 0)
        res.assert_called_once_with([1], 'expired', reason='flush_failed')

    def test_reconstruct_bypasses_cached_price(self):
        """A stale cached close must be dropped before re-pricing."""
        import pandas as pd
        self.engine._ohlcv_cache['AAPL'] = pd.DataFrame(
            {'close': [50.0, 50.0, 50.0, 50.0]})
        with patch('trading_engine.data_provider') as dp:
            dp.get_single_stock_bars.return_value = pd.DataFrame(
                {'close': [101.0, 101.0, 101.0, 102.0]})
            opp = self.engine._reconstruct_opportunity(_queued_signal())
        self.assertEqual(opp.entry_price, 102.0)

    def test_friday_evening_signal_fresh_after_monday_holiday(self):
        # Fri 2026-09-04 17:00 ET; Mon 09-07 is Labor Day; Tue 09-08 opens.
        queued = ET.localize(datetime(2026, 9, 4, 17, 0))
        now = ET.localize(datetime(2026, 9, 8, 9, 35)).astimezone(timezone.utc)
        self.assertFalse(TradingEngine._signal_is_stale(queued, now, 1))

    def test_friday_evening_signal_stale_by_wednesday(self):
        queued = ET.localize(datetime(2026, 10, 9, 17, 0))
        now = ET.localize(datetime(2026, 10, 14, 9, 35)).astimezone(
            timezone.utc)
        self.assertTrue(TradingEngine._signal_is_stale(queued, now, 1))

    def test_friday_evening_signal_fresh_monday_normal_week(self):
        queued = ET.localize(datetime(2026, 10, 9, 17, 0))
        now = ET.localize(datetime(2026, 10, 12, 9, 35)).astimezone(
            timezone.utc)
        self.assertFalse(TradingEngine._signal_is_stale(queued, now, 1))

    def test_default_max_age_is_one_day(self):
        from config import globalConfig
        self.assertEqual(globalConfig.PENDING_SIGNAL_MAX_AGE_DAYS, 1)

    def test_flush_no_queue_is_noop(self):
        with patch('trading_engine.is_regular_trading_hours',
                   return_value=True), \
                patch('trading_engine.storage.load_pending_signals',
                      return_value=[]):
            summary = self.engine.flush_queued_entries()
        self.assertEqual(summary['flushed'], 0)


class TestSessionFlushHook(_EngineFixture):
    """execute_trading_session triggers a queue flush when RTH entries are
    enforced (the flush itself is RTH-guarded, so off-hours calls no-op)."""

    def _run_session(self):
        with patch.object(self.engine, '_positions_manager') as pm, \
                patch.object(self.engine, 'update_portfolio_orders'), \
                patch.object(self.engine, 'flush_queued_entries') as flush, \
                patch('trading_engine.storage') as st:
            pm.get_and_reconcile_positions.return_value = []
            st.save_positions = Mock()
            st.save_orders = Mock()
            st.load_pending_signals = Mock(return_value=[])
            self.engine.flush_queued_entries.return_value = {
                'flushed': 0}
            self.engine.execute_trading_session([])
        return flush

    def test_session_flushes_when_enforcement_on(self):
        flush = self._run_session()
        flush.assert_called_once_with()

    def test_session_skips_flush_when_enforcement_off(self):
        self.engine.enforce_rth_entries = False
        flush = self._run_session()
        flush.assert_not_called()


if __name__ == '__main__':
    unittest.main()
