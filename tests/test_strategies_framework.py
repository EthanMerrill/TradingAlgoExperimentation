#!/usr/bin/env python3
"""
Unit tests for the multi-strategy framework: registry, Strategy interface,
and the generic optimizer path (Phase B).
"""
import os
import sys
import unittest
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import patch

import numpy as np
import pandas as pd

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from strategies.base import BacktestResult, Strategy  # noqa: E402
from strategies.registry import (  # noqa: E402
    STRATEGY_REGISTRY,
    get_strategy,
    list_strategies,
    register,
)
from strategies.rsi import RSIStrategy  # noqa: E402
from optimizer import StrategyOptimizer  # noqa: E402


class _FakeStrategy(Strategy):
    """Deterministic test strategy: return is proportional to a 'quality' param."""

    name = "fake_strategy"

    def __init__(self, base_return: float = 0.05):
        self.base_return = base_return

    def get_param_grid(self, direction: str = "long") -> List[Dict[str, Any]]:
        return [
            {"quality": 1, "direction": direction},
            {"quality": 2, "direction": direction},
            {"quality": 3, "direction": direction},
        ]

    def backtest(self, data, symbol, initial_cash=10000, prepared=None, **params):
        quality = params.get("quality", 1)
        total_return = self.base_return * quality
        return BacktestResult(
            symbol=symbol,
            rsi_period=14, rsi_lower=30, rsi_upper=70,
            total_return=total_return,
            buy_and_hold_return=0.02,
            alpha=total_return - 0.02,
            num_trades=5, win_rate=0.7, avg_trade_duration=3.0,
            max_drawdown=0.05, sharpe_ratio=1.0 + quality * 0.5,
            calmar_ratio=1.0, profitable=total_return > 0,
            direction=params.get("direction", "long"),
            strategy_name=self.name,
            params=dict(params),
        )


def _make_data(n: int = 60) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    close = np.linspace(100, 110, n)
    return pd.DataFrame({"close": close}, index=idx)


class TestRegistry(unittest.TestCase):
    """Strategy registry behavior."""

    def test_rsi_registered(self):
        self.assertIn("rsi_mean_reversion", STRATEGY_REGISTRY)
        self.assertIs(STRATEGY_REGISTRY["rsi_mean_reversion"], RSIStrategy)

    def test_list_and_get(self):
        names = list_strategies()
        self.assertIn("rsi_mean_reversion", names)
        self.assertIs(get_strategy("rsi_mean_reversion"), RSIStrategy)

    def test_unknown_strategy_raises(self):
        with self.assertRaises(ValueError):
            get_strategy("no_such_strategy")

    def test_duplicate_registration_raises(self):
        class _Dup(Strategy):
            name = "rsi_mean_reversion"  # clash with existing

            def backtest(self, data, symbol, initial_cash=10000, prepared=None, **params):
                raise NotImplementedError

        with self.assertRaises(ValueError):
            register(_Dup)


class TestGenericOptimize(unittest.TestCase):
    """The default Strategy.optimize + StrategyOptimizer delegation path."""

    def test_optimize_picks_best_profitable(self):
        strategy = _FakeStrategy()
        data = _make_data()
        result = strategy.optimize(data, "FAKE", "long", 10000)
        self.assertIsNotNone(result)
        # quality=3 yields the highest return/score
        self.assertEqual(result.params["quality"], 3)
        self.assertGreater(result.composite_score, 0.0)

    def test_optimizer_delegates_to_strategy(self):
        strategy = _FakeStrategy()
        optimizer = StrategyOptimizer(strategy=strategy)
        with patch("optimizer.data_provider") as mock_dp:
            mock_dp.get_single_stock_bars.return_value = _make_data()
            result = optimizer.optimize_symbol(
                "FAKE", datetime(2025, 1, 1), datetime(2025, 6, 1), "long")
        self.assertIsNotNone(result)
        self.assertEqual(result.strategy_name, "fake_strategy")
        self.assertEqual(result.params["quality"], 3)

    def test_optimize_no_data_returns_none(self):
        optimizer = StrategyOptimizer(strategy=_FakeStrategy())
        with patch("optimizer.data_provider") as mock_dp:
            mock_dp.get_single_stock_bars.return_value = pd.DataFrame()
            result = optimizer.optimize_symbol(
                "FAKE", datetime(2025, 1, 1), datetime(2025, 6, 1), "long")
        self.assertIsNone(result)


class TestRsiParity(unittest.TestCase):
    """The RSI path must behave exactly as before the framework move."""

    def test_rsi_backtest_direct_and_parametrized_agree(self):
        data = _make_data(120)
        direct = RSIStrategy(14, 30, 70).backtest(data, "AAPL", 10000)
        parametrized = RSIStrategy(14, 30, 70).backtest(
            data, "AAPL", 10000, rsi_period=14, rsi_lower=30, rsi_upper=70,
            direction="long")
        self.assertEqual(direct.total_return, parametrized.total_return)
        self.assertEqual(direct.sharpe_ratio, parametrized.sharpe_ratio)
        self.assertEqual(direct.params["rsi_period"], 14)
        self.assertEqual(direct.strategy_name, "rsi_mean_reversion")

    def test_build_consolidated_trades_alias(self):
        strategy = RSIStrategy(14, 30, 70)
        trade_details = [{
            "entry_date": pd.Timestamp("2026-01-15"),
            "exit_date": pd.Timestamp("2026-01-20"),
            "entry_price": 150.0,
            "exit_price": 155.0,
            "return": 0.0333,
            "duration": 5,
            "exit_reason": "rsi_cross",
            "direction": "long",
        }]
        result = BacktestResult(
            symbol="AAPL", rsi_period=14, rsi_lower=30, rsi_upper=70,
            total_return=0.15, buy_and_hold_return=0.10, alpha=0.05,
            num_trades=1, win_rate=1.0, avg_trade_duration=5.0,
            max_drawdown=0.03, sharpe_ratio=1.5, calmar_ratio=2.0,
            profitable=True, trade_details=trade_details,
        )
        df = strategy.build_consolidated_trades([result])
        df_legacy = strategy.build_consolidated_trades_df([result])
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["symbol"], "AAPL")
        self.assertEqual(df.iloc[0]["exit_reason"], "rsi_cross")
        pd.testing.assert_frame_equal(df, df_legacy)


if __name__ == "__main__":
    unittest.main()
