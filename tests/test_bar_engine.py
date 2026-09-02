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

    def test_df_row_round_trip(self):
        pos = Position(
            symbol="AAA", quantity=10.0, entry_price=10.0,
            current_price=10.5, current_rsi=0.0,
            entry_date=datetime.now(), alpha=0.05,
            rsi_period=14, rsi_lower=30, rsi_upper=70,
            intraday=True,
        )
        row = pd.Series(normalize_position_for_save(pos))
        manager = PositionsManager(Mock(), Mock())
        restored = manager._df_row_to_position(row)
        self.assertIs(restored.intraday, True)

    def test_legacy_row_defaults_false(self):
        row = pd.Series({
            "symbol": "AAA", "shares": 10.0, "entry_price": 10.0,
            "current_price": 10.5, "current_rsi": 0.0,
            "entry_date": pd.Timestamp.now(), "rsi_period": 14,
            "rsi_lower": 30, "rsi_upper": 70, "alpha": 0.05,
            "closed": False, "exit_date": None,
        })
        manager = PositionsManager(Mock(), Mock())
        restored = manager._df_row_to_position(row)
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
        self.engine.place_market_sell_order = Mock(return_value=True)
        self.positions_manager.close_position = Mock()

        summary = self.bar_engine.close_intraday_positions()
        self.assertEqual(summary["positions_exited"], 1)
        self.engine.place_market_sell_order.assert_called_once()
        self.assertEqual(
            self.engine.place_market_sell_order.call_args[0][0], "AAA")
        self.positions_manager.close_position.assert_called_once_with("AAA")

    def test_close_dry_run_does_not_persist(self):
        self.bar_engine.set_dry_run_mode(True)
        intraday_pos = Mock()
        intraday_pos.closed = False
        intraday_pos.intraday = True
        intraday_pos.side = "long"
        intraday_pos.symbol = "AAA"
        intraday_pos.quantity = 10.0
        self.positions_manager.positions = [intraday_pos]
        self.engine.place_market_sell_order = Mock(return_value=True)
        self.positions_manager.close_position = Mock()

        summary = self.bar_engine.close_intraday_positions()
        self.assertEqual(summary["positions_exited"], 1)
        self.positions_manager.close_position.assert_not_called()
        self.positions_manager.persist_positions.assert_not_called()


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
        engine.calculate_todays_stop_loss_and_take_profit = Mock(
            return_value=(9.0, 11.0))
        engine.place_oco_close_order = Mock(return_value=True)

        summary = {"positions_exited": 0, "orders_placed": 0}
        result = engine.update_portfolio_orders(summary, [intraday_pos])
        self.assertEqual(result["positions_exited"], 0)
        self.assertEqual(result["orders_placed"], 0)
        engine.place_oco_close_order.assert_not_called()
        engine.place_market_sell_order.assert_not_called()


if __name__ == "__main__":
    unittest.main()
