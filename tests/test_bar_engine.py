#!/usr/bin/env python3
"""
Phase D tests: bar-loop engine (intraday strategies) + timeframe parameterization
+ intraday position flag/lifecycle.
"""
import os
import sys
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
import pytz

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from bar_engine import BarLoopEngine  # noqa: E402
from data_provider import _parse_timeframe  # noqa: E402
from positions import Position, PositionsManager  # noqa: E402
from strategies.base import LiveSignal, Strategy  # noqa: E402
from strategies.registry import register  # noqa: E402
from storage.backend import normalize_position_for_save  # noqa: E402
from trading_engine import TradingOpportunity  # noqa: E402
from alpaca.data.timeframe import TimeFrameUnit  # noqa: E402

US_EASTERN = pytz.timezone("US/Eastern")


class _BarLoopFake(Strategy):
    """Registered bar_loop strategy emitting a fixed long signal."""
    name = "test_bar_loop"
    execution_style = "bar_loop"
    bar_size = "5m"

    def __init__(self):
        self.last_ctx = None

    def backtest(self, data, symbol, initial_cash=10000, prepared=None, **params):
        raise NotImplementedError

    def evaluate_live_signals(self, ctx):
        self.last_ctx = ctx
        return [LiveSignal(
            symbol="AAA", direction="long", entry_price=10.0,
            stop_loss=9.5, take_profit=11.0,
            backtest_return=0.2, alpha=0.05, win_rate=0.9,
            composite_score=4.0, num_trades=8, strategy_name=self.name,
        )]


class _SessionFake(Strategy):
    """Registered session strategy (should NOT be traded by the bar loop)."""
    name = "test_session_fake"
    execution_style = "session"

    def backtest(self, data, symbol, initial_cash=10000, prepared=None, **params):
        raise NotImplementedError


class TestTimeframeParsing(unittest.TestCase):
    def _check(self, tf, amount, unit):
        self.assertEqual(tf.amount, amount)
        self.assertEqual(tf.unit, unit)

    def test_parse_5m(self):
        self._check(_parse_timeframe("5m"), 5, TimeFrameUnit.Minute)

    def test_parse_1h(self):
        self._check(_parse_timeframe("1h"), 1, TimeFrameUnit.Hour)

    def test_parse_1d(self):
        self._check(_parse_timeframe("1d"), 1, TimeFrameUnit.Day)

    def test_parse_invalid_raises(self):
        with self.assertRaises(ValueError):
            _parse_timeframe("banana")


class TestIntradayPositionFlag(unittest.TestCase):
    def test_normalize_emits_intraday(self):
        pos = Position(
            symbol="AAA", quantity=10.0, entry_price=10.0,
            current_price=10.5, current_rsi=0.0,
            entry_date=datetime.now(), alpha=0.05,
            rsi_period=14, rsi_lower=30, rsi_upper=70,
            intraday=True,
        )
        d = normalize_position_for_save(pos)
        self.assertIs(d["intraday"], True)
        self.assertEqual(d["strategy_name"], "rsi_mean_reversion")

    def test_metadata_round_trip(self):
        """intraday survives metadata row -> broker join."""
        pos = Position(
            symbol="AAA", quantity=10.0, entry_price=10.0,
            current_price=10.5, current_rsi=0.0,
            entry_date=datetime.now(), alpha=0.05,
            rsi_period=14, rsi_lower=30, rsi_upper=70,
            intraday=True,
        )
        meta = PositionsManager._metadata_row(pos)
        broker_row = pd.Series({"symbol": "AAA", "qty": 10.0,
                                "avg_entry_price": 10.0,
                                "current_price": 10.5})
        restored = PositionsManager._position_from_broker(broker_row, meta)
        self.assertIs(restored.intraday, True)

    def test_missing_metadata_defaults_false(self):
        broker_row = pd.Series({"symbol": "AAA", "qty": 10.0,
                                "avg_entry_price": 10.0,
                                "current_price": 10.5})
        restored = PositionsManager._position_from_broker(broker_row, {})
        self.assertIs(restored.intraday, False)


class TestBarLoopEngine(unittest.TestCase):
    def setUp(self):
        register(_BarLoopFake)
        register(_SessionFake)

        self.cfg = SimpleNamespace(
            STRATEGIES_ENABLED=["test_bar_loop", "test_session_fake"],
            ENABLE_SHORT_SELLING=False,
        )
        self.cfg_patcher = patch("bar_engine.globalConfig", self.cfg)
        self.cfg_patcher.start()
        self.addCleanup(self.cfg_patcher.stop)

        self.engine = Mock()
        self.engine._ohlcv_cache = {}
        self.positions_manager = Mock()
        self.positions_manager.positions = []
        self.bar_engine = BarLoopEngine(self.engine, self.positions_manager)

    def _result(self, strategy_name="test_bar_loop"):
        r = Mock()
        r.strategy_name = strategy_name
        return r

    def test_enabled_bar_loop_strategies_filters(self):
        strategies = self.bar_engine.enabled_bar_loop_strategies()
        names = [s.name for s in strategies]
        self.assertEqual(names, ["test_bar_loop"])  # session fake excluded

    def test_enabled_bar_loop_strategies_reuse_instances(self):
        # Same instance across cycles => per-session dedupe state survives.
        first = self.bar_engine.enabled_bar_loop_strategies()
        second = self.bar_engine.enabled_bar_loop_strategies()
        self.assertEqual(len(first), 1)
        self.assertIs(first[0], second[0])

    def test_is_rth_and_session_ended(self):
        open_dt = US_EASTERN.localize(
            datetime(2026, 9, 1, 11, 0))  # Tue 11:00 ET
        self.assertTrue(self.bar_engine.is_rth(open_dt))
        self.assertFalse(self.bar_engine.session_ended(open_dt))

        closed_dt = US_EASTERN.localize(datetime(2026, 9, 1, 16, 30))
        self.assertFalse(self.bar_engine.is_rth(closed_dt))
        self.assertTrue(self.bar_engine.session_ended(closed_dt))

        weekend = US_EASTERN.localize(datetime(2026, 9, 5, 11, 0))  # Sat
        self.assertFalse(self.bar_engine.is_rth(weekend))

    def test_run_intraday_cycle_dispatches_signals(self):
        self.engine._execute_purchases = Mock()
        self.engine._execute_shorts = Mock()
        canned = TradingOpportunity(
            symbol="AAA", current_rsi=0.0, target_rsi_lower=0,
            target_rsi_upper=0, rsi_period=14, backtest_return=0.2,
            alpha=0.05, win_rate=0.9, entry_price=10.0,
            stop_loss_price=9.5, take_profit_price=11.0,
            num_trades=8, composite_score=4.0, direction="long",
            strategy_name="test_bar_loop", intraday=True,
        )
        self.engine._signals_to_opportunities = Mock(
            return_value=[canned])
        results = [self._result("test_bar_loop"),
                   self._result("rsi_mean_reversion")]
        summary = self.bar_engine.run_intraday_cycle(results)
        self.assertEqual(summary["signals"], 1)
        self.engine._execute_purchases.assert_called_once()
        long_opps = self.engine._execute_purchases.call_args[0][1]
        self.assertEqual(long_opps, [canned])
        self.assertIs(long_opps[0].intraday, True)
        self.engine._execute_shorts.assert_not_called()

    def test_strategy_receives_its_own_results(self):
        strategy = _BarLoopFake()
        self.engine._signals_to_opportunities = Mock(return_value=[])
        self.engine._execute_purchases = Mock()
        with patch.object(self.bar_engine, "enabled_bar_loop_strategies",
                          return_value=[strategy]):
            results = [self._result("test_bar_loop"), self._result("other")]
            self.bar_engine.run_intraday_cycle(results)
        self.assertIsNotNone(strategy.last_ctx)
        self.assertEqual(
            [r.strategy_name for r in strategy.last_ctx.strategy_results],
            ["test_bar_loop"])

    def test_strategy_evaluated_without_backtest_results(self):
        # Bar-loop strategies compute signals from fresh data, so they must be
        # evaluated even when the nightly backtest produced nothing for them.
        strategy = _BarLoopFake()
        self.engine._signals_to_opportunities = Mock(return_value=[])
        self.engine._execute_purchases = Mock()
        with patch.object(self.bar_engine, "enabled_bar_loop_strategies",
                          return_value=[strategy]):
            self.bar_engine.run_intraday_cycle([])
            self.assertIsNotNone(strategy.last_ctx)
            self.assertEqual(strategy.last_ctx.strategy_results, [])

            strategy.last_ctx = None
            self.bar_engine.run_intraday_cycle([self._result("other")])
            self.assertIsNotNone(strategy.last_ctx)
            self.assertEqual(strategy.last_ctx.strategy_results, [])

    def test_close_intraday_positions_only_intraday(self):
        intraday_pos = Mock()
        intraday_pos.closed = False
        intraday_pos.intraday = True
        intraday_pos.side = "long"
        intraday_pos.symbol = "AAA"
        intraday_pos.quantity = 10.0
        daily_pos = Mock()
        daily_pos.closed = False
        daily_pos.intraday = False
        daily_pos.side = "long"
        daily_pos.symbol = "BBB"
        daily_pos.quantity = 5.0
        self.positions_manager.positions = [intraday_pos, daily_pos]
        self.engine.close_position_at_market = Mock(return_value=True)
        self.positions_manager.close_position = Mock()

        summary = self.bar_engine.close_intraday_positions()
        self.assertEqual(summary["positions_exited"], 1)
        self.engine.close_position_at_market.assert_called_once()
        self.assertEqual(
            self.engine.close_position_at_market.call_args[0][0].symbol, "AAA")
        self.assertEqual(
            self.engine.close_position_at_market.call_args[0][1],
            "intraday_session_close")

    def test_close_dry_run_does_not_persist(self):
        self.bar_engine.set_dry_run_mode(True)
        intraday_pos = Mock()
        intraday_pos.closed = False
        intraday_pos.intraday = True
        intraday_pos.side = "long"
        intraday_pos.symbol = "AAA"
        intraday_pos.quantity = 10.0
        self.positions_manager.positions = [intraday_pos]
        # In dry-run the engine reports success without touching the broker,
        # so the position is counted as exited but never persisted.
        self.engine.close_position_at_market = Mock(return_value=True)
        self.positions_manager.close_position = Mock()

        summary = self.bar_engine.close_intraday_positions()
        self.assertEqual(summary["positions_exited"], 1)
        self.positions_manager.close_position.assert_not_called()
        self.positions_manager.persist_positions.assert_not_called()


class TestIntradayExit(unittest.TestCase):
    def setUp(self):
        self.cfg = SimpleNamespace(INTRADAY_EXIT_TIME="15:55",
                                   STRATEGIES_ENABLED=[])
        p = patch("bar_engine.globalConfig", self.cfg)
        p.start()
        self.addCleanup(p.stop)
        self.engine = Mock()
        self.pm = Mock()
        self.be = BarLoopEngine(self.engine, self.pm)

    def _pos(self, sym="AAA", qty=-10.0):
        pos = Mock(closed=False, intraday=True, symbol=sym, quantity=qty)
        return pos

    def test_exit_due_before_close_only_during_rth(self):
        from utils.calendar import US_EASTERN
        wed = lambda h, m: US_EASTERN.localize(datetime(2026, 9, 2, h, m))
        self.assertFalse(self.be.intraday_exit_due(wed(15, 54)))
        self.assertTrue(self.be.intraday_exit_due(wed(15, 55)))
        self.assertFalse(self.be.intraday_exit_due(wed(16, 30)))  # post close

    def test_exit_due_before_early_close(self):
        # Day after Thanksgiving 2026 closes at 13:00 ET → cutoff 12:55.
        from utils.calendar import US_EASTERN
        fri = lambda h, m: US_EASTERN.localize(datetime(2026, 11, 27, h, m))
        self.assertFalse(self.be.intraday_exit_due(fri(12, 54)))
        self.assertTrue(self.be.intraday_exit_due(fri(12, 55)))
        self.assertFalse(self.be.intraday_exit_due(fri(13, 30)))  # post close

    def test_release_called_before_close_with_abs_qty(self):
        order = Mock()
        order.attach_mock(self.engine._release_shares_for_exit, "release")
        order.attach_mock(self.engine.close_position_at_market, "close")
        self.engine.close_position_at_market.return_value = True
        self.pm.positions = [self._pos("AAA", -10.0)]
        self.be.close_intraday_positions()
        self.assertEqual([c[0] for c in order.mock_calls], ["release", "close"])
        self.engine._release_shares_for_exit.assert_called_once_with("AAA", 10.0)

    def test_failed_close_reported(self):
        self.engine.close_position_at_market.return_value = False
        self.pm.positions = [self._pos()]
        summary = self.be.close_intraday_positions()
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["positions_exited"], 0)


class TestUpdatePortfolioOrdersSkipsIntraday(unittest.TestCase):
    def test_intraday_positions_skipped(self):
        with patch("trading_engine.data_provider"):
            from trading_engine import TradingEngine
            engine = TradingEngine()
        engine._positions_manager = Mock()
        intraday_pos = Mock()
        intraday_pos.intraday = True
        intraday_pos.symbol = "AAA"
        intraday_pos.entry_date = datetime(2026, 9, 1)
        intraday_pos.quantity = 10.0
        intraday_pos.side = "long"
        engine.place_market_sell_order = Mock(return_value=True)
        engine.close_position_at_market = Mock(return_value=True)
        engine.calculate_todays_stop_loss_and_take_profit = Mock(
            return_value=(9.0, 11.0))
        engine.place_oco_close_order = Mock(return_value=True)

        summary = {"positions_exited": 0, "orders_placed": 0}
        result = engine.update_portfolio_orders(summary, [intraday_pos])
        self.assertEqual(result["positions_exited"], 0)
        self.assertEqual(result["orders_placed"], 0)
        engine.place_oco_close_order.assert_not_called()
        engine.place_market_sell_order.assert_not_called()
        engine.close_position_at_market.assert_not_called()


if __name__ == "__main__":
    unittest.main()
