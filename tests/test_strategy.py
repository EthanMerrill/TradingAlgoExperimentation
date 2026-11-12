#!/usr/bin/env python3
"""
Unit tests for the strategy module.
"""
import os
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from strategies.base import BacktestResult  # noqa: E402
from strategies.rsi import RSIStrategy  # noqa: E402


class TestBacktestResult(unittest.TestCase):
    """Test cases for the BacktestResult dataclass."""

    def test_backtest_result_creation(self):
        """Test creating a BacktestResult object."""
        result = BacktestResult(
            symbol="AAPL",
            total_return=0.15,
            buy_and_hold_return=0.10,
            alpha=0.05,
            num_trades=5,
            win_rate=0.6,
            avg_trade_duration=10.5,
            max_drawdown=0.08,
            sharpe_ratio=1.2,
            profitable=True,
            params={"rsi_period": 14, "rsi_lower": 30,
                    "rsi_upper": 70, "current_rsi": 45.0},
        )

        self.assertEqual(result.params.get("rsi_period"), 14)
        self.assertEqual(result.params.get("rsi_lower"), 30)
        self.assertEqual(result.params.get("rsi_upper"), 70)
        self.assertEqual(result.total_return, 0.15)
        self.assertEqual(result.alpha, 0.05)
        self.assertTrue(result.profitable)
        self.assertEqual(result.params.get("current_rsi"), 45.0)

    def test_backtest_result_optional_fields(self):
        """Test BacktestResult with optional fields."""
        result = BacktestResult(
            symbol="TSLA",
            params={"rsi_period": 21, "rsi_lower": 25, "rsi_upper": 75},
            total_return=0.20,
            buy_and_hold_return=0.15,
            alpha=0.05,
            num_trades=3,
            win_rate=0.67,
            avg_trade_duration=15.0,
            max_drawdown=0.12,
            sharpe_ratio=0.9,
            profitable=True
        )

        self.assertIsNone(result.params.get("current_rsi"))
        self.assertIsNone(result.trade_details)


class TestRSIStrategy(unittest.TestCase):
    """Test cases for the RSIStrategy class."""

    def setUp(self):
        """Set up test fixtures."""
        self.strategy = RSIStrategy(
            rsi_period=14, rsi_lower=30, rsi_upper=70, max_hold_days=30)

    def test_rsi_strategy_init(self):
        """Test RSIStrategy initialization."""
        self.assertEqual(self.strategy.rsi_period, 14)
        self.assertEqual(self.strategy.rsi_lower, 30)
        self.assertEqual(self.strategy.rsi_upper, 70)
        self.assertEqual(self.strategy.max_hold_days, 30)

    @patch('strategies.rsi.globalConfig')
    def test_rsi_strategy_init_with_config_max_hold_days(self, mock_config):
        """Test RSIStrategy initialization using globalConfig for max_hold_days."""
        mock_config.MAX_HOLD_DAYS = 45

        strategy = RSIStrategy(rsi_period=21, rsi_lower=25, rsi_upper=75)

        self.assertEqual(strategy.max_hold_days, 45)

    def test_calculate_rsi(self):
        """Test RSI calculation via TechnicalIndicators."""
        from indicators import TechnicalIndicators
        # Create sample price data
        prices = pd.Series([100, 105, 103, 108, 110, 107,
                           112, 115, 113, 118, 120, 117, 122, 125, 123, 128])
        df = pd.DataFrame({'close': prices})

        rsi = TechnicalIndicators.calculate_rsi(df, self.strategy.rsi_period)

        self.assertIsInstance(rsi, pd.Series)
        self.assertEqual(len(rsi), len(prices))
        # Check that RSI values are within valid range (0-100)
        valid_rsi = rsi.dropna()
        self.assertTrue(all(0 <= val <= 100 for val in valid_rsi))

    def test_simulate_trades_long(self):
        """Long: RSI-cross entry at next open, trade list + equity are consistent."""
        dates = pd.date_range('2024-01-01', periods=13, freq='D')
        rsi_data = pd.Series(
            [70, 75, 80, 60, 25, 20, 35, 45, 75, 80, 25, 15, 40], index=dates)
        prices = pd.Series([100, 105, 110, 108, 95, 90,
                           98, 102, 110, 115, 95, 85, 95], index=dates)
        df = pd.DataFrame({'close': prices})

        trades, equity = self.strategy._simulate_trades(df, rsi_data, 10000.0)

        self.assertGreater(len(trades), 0)
        self.assertEqual(len(equity), len(df))
        for t in trades:
            self.assertIsNotNone(t['exit_reason'])
            for key in ('entry_price', 'exit_price', 'entry_date', 'exit_date',
                        'return', 'return_pct', 'side', 'duration', 'direction'):
                self.assertIn(key, t)
        # First entry: signal on bar 4 close (RSI 70->... cross below 30 at idx 4)
        self.assertEqual(trades[0]['entry_date'], dates[5])

    def test_short_simulate_trades_cross_above_entry(self):
        """Short signals: RSI cross-above rsi_upper fires entry, filled next bar."""
        strategy = RSIStrategy(
            rsi_period=14, rsi_lower=30, rsi_upper=70, direction="short")

        dates = pd.date_range('2024-01-01', periods=12, freq='D')
        # RSI crosses above 70 at index 4 (prev=68, curr=75)
        rsi_data = pd.Series(
            [50, 55, 60, 68, 75, 80, 78, 72, 65, 55, 45, 35], index=dates)
        prices = pd.Series(
            [100, 102, 104, 106, 108, 110, 109, 107, 105, 103, 101, 99], index=dates)

        df = pd.DataFrame({'close': prices})
        trades, equity = strategy._simulate_trades(df, rsi_data, 10000.0)

        # Cover on RSI cross-below 30 never happens; position is open at the
        # end (marked to market, not reported), so verify via equity: the
        # short entered at bar 5 and the price then falls -> equity rises.
        self.assertGreater(equity[-1], 10000.0)
        self.assertEqual(equity[4], 10000.0)  # flat through signal bar

    def test_short_returns_profitable_on_price_drop(self):
        """A price drop after a short entry yields a positive trade and equity."""
        strategy = RSIStrategy(
            rsi_period=14, rsi_lower=30, rsi_upper=70, direction="short")

        dates = pd.date_range('2024-01-01', periods=20, freq='D')
        prices = [100] * 20
        rsi_vals = [50] * 20

        # RSI cross-above 70 at index 5 -> short entry (fills bar 6 open)
        rsi_vals[4] = 68
        rsi_vals[5] = 75
        prices[6] = 95
        prices[7] = 90
        prices[8] = 90
        # RSI cross-below 30 at index 9 -> cover (fills bar 10 open)
        rsi_vals[8] = 35
        rsi_vals[9] = 25
        prices[9] = 85
        prices[10] = 85

        df = pd.DataFrame({'close': pd.Series(prices, index=dates)})
        rsi = pd.Series(rsi_vals, index=dates)

        trades, equity = strategy._simulate_trades(df, rsi, 10000.0)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]['exit_reason'], 'rsi_cross')
        self.assertGreater(trades[0]['return'], 0)
        self.assertGreater(equity[-1], 10000.0,
                           f"Short should profit on price drop; final={equity[-1]:.2f}")

    def test_backtest_with_insufficient_data(self):
        """Test backtest with insufficient data returns a null result, not None."""
        # Create insufficient data (less than RSI period)
        prices = pd.Series([100, 105, 103])
        df = pd.DataFrame({'close': prices})

        result = self.strategy.backtest(df, "TEST")

        self.assertIsInstance(result, BacktestResult)
        self.assertFalse(result.profitable)
        self.assertEqual(result.num_trades, 0)

    def test_backtest_with_valid_data(self):
        """Test backtest with valid data."""
        # Create sufficient sample data
        np.random.seed(42)  # For reproducible results
        prices = pd.Series(100 + np.cumsum(np.random.randn(100) * 0.02))
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        df = pd.DataFrame({'close': prices}, index=dates)

        result = self.strategy.backtest(df, "TEST")

        self.assertIsInstance(result, BacktestResult)
        self.assertEqual(result.symbol, "TEST")
        self.assertEqual(result.params.get("rsi_period"), 14)
        self.assertEqual(result.params.get("rsi_lower"), 30)
        self.assertEqual(result.params.get("rsi_upper"), 70)

    def test_strategy_direction_defaults_to_long(self):
        """Test that strategy direction defaults to 'long'."""
        strategy = RSIStrategy(rsi_period=14, rsi_lower=30, rsi_upper=70)
        self.assertEqual(strategy.direction, "long")


if __name__ == '__main__':
    unittest.main()
