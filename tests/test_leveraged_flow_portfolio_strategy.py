#!/usr/bin/env python3
"""
Tests for the live leveraged-ETF flow portfolio strategy.
"""
import os
import sys
import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from strategies.leveraged_flow_portfolio_strategy import (  # noqa: E402
    LeveragedFlowPortfolioStrategy,
)
from strategies.leveraged_rebalance_signal import prepare_features  # noqa: E402
from strategies.registry import get_strategy, list_strategies  # noqa: E402
from strategies.base import StrategyContext  # noqa: E402

_BARS_PER_DAY = 78


def _bars(days=6, base=100.0, rets=None):
    """Gap-at-open bars: the day's return is visible from the open."""
    idx_rets = rets if rets is not None else [0.0] * days
    sessions = pd.bdate_range("2026-09-14", periods=days)
    frames = []
    level = base
    for i, day in enumerate(sessions):
        start = (pd.Timestamp(day).tz_localize("US/Eastern")
                 + pd.Timedelta(hours=9, minutes=30))
        idx = pd.date_range(start, periods=_BARS_PER_DAY, freq="5min")
        n = len(idx)
        level = level * (1.0 + idx_rets[i % len(idx_rets)])
        close = np.full(n, level)
        frames.append(pd.DataFrame({
            "open": close, "high": close * 1.0005, "low": close * 0.9995,
            "close": close, "volume": np.full(n, 1000.0),
        }, index=idx))
    return pd.concat(frames)


def _ctx(as_of, bars_map):
    provider = Mock()
    provider.get_single_stock_bars = Mock(
        side_effect=lambda symbol, s, e, **kw: bars_map.get(symbol, pd.DataFrame()))
    return StrategyContext(
        data_provider=provider, config=SimpleNamespace(),
        as_of=as_of, ohlcv_cache={})


class TestRegistration(unittest.TestCase):
    def test_registered(self):
        self.assertIn("leveraged_flow_portfolio", list_strategies())
        self.assertIs(
            get_strategy("leveraged_flow_portfolio"),
            LeveragedFlowPortfolioStrategy)

    def test_metadata(self):
        strategy = LeveragedFlowPortfolioStrategy()
        self.assertEqual(strategy.execution_style, "bar_loop")
        self.assertEqual(strategy.bar_size, "5m")
        self.assertEqual(strategy.data_timeframe, "5m")
        self.assertIn("NVDA", strategy.symbol_universe())


class TestBacktest(unittest.TestCase):
    def test_per_symbol_positive_flow_trades(self):
        strategy = LeveragedFlowPortfolioStrategy(min_abs_notional=0.0)
        # Gap-at-open bars make entry==close, so per-trade returns are 0; the
        # assertion here is that positive-flow sessions are TRADED (counted),
        # which is the activation path the engine needs.
        bars = _bars(days=8, rets=[0.0, 0.02, 0.0, 0.02, 0.0, 0.02, 0.0, 0.02])
        result = strategy.backtest(bars, "TSLA")
        self.assertEqual(result.strategy_name, "leveraged_flow_portfolio")
        self.assertEqual(result.direction, "long")
        self.assertEqual(result.num_trades, 4)

    def test_negative_flow_never_trades(self):
        strategy = LeveragedFlowPortfolioStrategy(min_abs_notional=0.0)
        bars = _bars(days=8, rets=[0.0, -0.02, 0.0, -0.02])
        result = strategy.backtest(bars, "TSLA")
        self.assertEqual(result.num_trades, 0)
        self.assertFalse(result.profitable)

    def test_min_flow_filter(self):
        strategy = LeveragedFlowPortfolioStrategy(min_abs_notional=1e12)
        bars = _bars(days=8, rets=[0.0, 0.02])
        result = strategy.backtest(bars, "TSLA")
        self.assertEqual(result.num_trades, 0)


class TestLiveSignals(unittest.TestCase):
    """evaluate_live_signals: dedupe, timing, selection, and price safety."""

    def _bars_map(self, up_rets=(0.02, 0.03, 0.02, 0.03, 0.02, 0.03)):
        # One strongly up underlying + ETF bars matching the map's funds.
        from strategies.leveraged_single_stock_etfs import (
            LEVERAGED_SINGLE_STOCK_ETFS)
        bars_map = {"TSLA": _bars(days=6, rets=list(up_rets))}
        for etf in LEVERAGED_SINGLE_STOCK_ETFS:
            fund = LEVERAGED_SINGLE_STOCK_ETFS[etf]
            if fund.underlying == "TSLA":
                bars_map[etf] = _bars(days=6, rets=list(up_rets))
            else:
                bars_map[etf] = _bars(days=6, rets=[0.0] * 6)
        return bars_map

    def test_no_signal_before_decision_time(self):
        strategy = LeveragedFlowPortfolioStrategy(entry_time="15:20")
        # 14:00 ET < 15:20 decision time.
        as_of = datetime(2026, 9, 16, 18, 0)   # 14:00 ET (UTC-4)
        signals = strategy.evaluate_live_signals(_ctx(as_of, self._bars_map()))
        self.assertEqual(signals, [])

    def test_signal_emitted_at_decision_time(self):
        strategy = LeveragedFlowPortfolioStrategy(entry_time="15:20")
        as_of = datetime(2026, 9, 16, 19, 30)  # 15:30 ET
        signals = strategy.evaluate_live_signals(_ctx(as_of, self._bars_map()))
        self.assertGreater(len(signals), 0)
        signal = signals[0]
        self.assertEqual(signal.direction, "long")
        self.assertEqual(signal.strategy_name, "leveraged_flow_portfolio")
        self.assertGreater(signal.entry_price, 0)
        self.assertIsNotNone(signal.stop_loss)
        self.assertIsNotNone(signal.take_profit)
        self.assertLess(signal.stop_loss, signal.entry_price)
        self.assertGreater(signal.take_profit, signal.entry_price)
        self.assertIn("flow_notional_usd", signal.extra)

    def test_dedupe_per_session(self):
        strategy = LeveragedFlowPortfolioStrategy(entry_time="15:20")
        bars_map = self._bars_map()
        as_of = datetime(2026, 9, 16, 19, 30)
        first = strategy.evaluate_live_signals(_ctx(as_of, bars_map))
        second = strategy.evaluate_live_signals(_ctx(as_of, bars_map))
        self.assertGreater(len(first), 0)
        self.assertEqual(second, [])

    def test_next_session_signals_again(self):
        strategy = LeveragedFlowPortfolioStrategy(entry_time="15:20")
        bars_map = self._bars_map()
        day1 = datetime(2026, 9, 16, 19, 30)
        day2 = datetime(2026, 9, 17, 19, 30)
        self.assertGreater(
            len(strategy.evaluate_live_signals(_ctx(day1, bars_map))), 0)
        self.assertGreater(
            len(strategy.evaluate_live_signals(_ctx(day2, bars_map))), 0)

    def test_weekend_no_signals(self):
        strategy = LeveragedFlowPortfolioStrategy(entry_time="15:20")
        saturday = datetime(2026, 9, 19, 19, 30)
        self.assertEqual(
            strategy.evaluate_live_signals(_ctx(saturday, self._bars_map())),
            [])

    def test_no_data_no_signals(self):
        strategy = LeveragedFlowPortfolioStrategy(entry_time="15:20")
        as_of = datetime(2026, 9, 16, 19, 30)
        signals = strategy.evaluate_live_signals(_ctx(as_of, {}))
        self.assertEqual(signals, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
