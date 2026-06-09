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

from strategy import BacktestResult, RSIStrategy  # noqa: E402


class TestBacktestResult(unittest.TestCase):
    """Test cases for the BacktestResult dataclass."""

    def test_backtest_result_creation(self):
        """Test creating a BacktestResult object."""
        result = BacktestResult(
            symbol="AAPL",
            rsi_period=14,
            rsi_lower=30,
            rsi_upper=70,
            total_return=0.15,
            buy_and_hold_return=0.10,
            alpha=0.05,
            num_trades=5,
            win_rate=0.6,
            avg_trade_duration=10.5,
            max_drawdown=0.08,
            sharpe_ratio=1.2,
            profitable=True,
            current_rsi=45.0
        )

        self.assertEqual(result.symbol, "AAPL")
        self.assertEqual(result.rsi_period, 14)
        self.assertEqual(result.rsi_lower, 30)
        self.assertEqual(result.rsi_upper, 70)
        self.assertEqual(result.total_return, 0.15)
        self.assertEqual(result.alpha, 0.05)
        self.assertTrue(result.profitable)
        self.assertEqual(result.current_rsi, 45.0)

    def test_backtest_result_optional_fields(self):
        """Test BacktestResult with optional fields."""
        result = BacktestResult(
            symbol="TSLA",
            rsi_period=21,
            rsi_lower=25,
            rsi_upper=75,
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

        self.assertIsNone(result.current_rsi)
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

    @patch('strategy.globalConfig')
    def test_rsi_strategy_init_with_config_max_hold_days(self, mock_config):
        """Test RSIStrategy initialization using globalConfig for max_hold_days."""
        mock_config.MAX_HOLD_DAYS = 45

        strategy = RSIStrategy(rsi_period=21, rsi_lower=25, rsi_upper=75)

        self.assertEqual(strategy.max_hold_days, 45)

    def test_calculate_rsi(self):
        """Test RSI calculation via TechnicalIndicators."""
        from data_provider import TechnicalIndicators
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

    def test_generate_signals(self):
        """Test signal generation."""
        # Create sample RSI data with proper datetime index
        dates = pd.date_range('2024-01-01', periods=13, freq='D')
        rsi_data = pd.Series(
            [70, 75, 80, 60, 25, 20, 35, 45, 75, 80, 25, 15, 40], index=dates)
        prices = pd.Series([100, 105, 110, 108, 95, 90,
                           98, 102, 110, 115, 95, 85, 95], index=dates)

        df = pd.DataFrame({'close': prices, 'rsi': rsi_data})

        signals = self.strategy._generate_signals(df, rsi_data)

        self.assertIn('buy_signal', signals.columns)
        self.assertIn('sell_signal', signals.columns)
        self.assertIn('sell_reason', signals.columns)

        # Check that we get buy signals when RSI is low
        buy_signals = signals[signals['buy_signal']]
        self.assertTrue(len(buy_signals) > 0)

        # Check that sell_reason is set for any sell signals
        sell_signals = signals[signals['sell_signal']]
        for idx in sell_signals.index:
            reason = signals.loc[idx, 'sell_reason']
            self.assertIsNotNone(reason, f"Sell at {idx} has no reason")

    def test_calculate_returns(self):
        """Test returns calculation."""
        # Create sample data with signals and proper datetime index
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        data = {
            'close': [100, 105, 110, 108, 95, 90, 98, 102, 110, 115],
            'buy_signal': [False, False, False, False, True, False, False, False, False, False],
            'sell_signal': [False, False, False, False, False, False, False, True, False, False],
            'position': [0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
        }
        df = pd.DataFrame(data, index=dates)

        returns = self.strategy._calculate_returns(df, pd.DataFrame({
            'buy_signal': data['buy_signal'],
            'sell_signal': data['sell_signal'],
            'position': data['position']
        }, index=dates), 10000.0)

        self.assertIn('portfolio_value', returns.columns)
        self.assertIn('daily_returns', returns.columns)

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
        self.assertEqual(result.rsi_period, 14)
        self.assertEqual(result.rsi_lower, 30)
        self.assertEqual(result.rsi_upper, 70)

    def test_short_generate_signals_cross_above_entry(self):
        """Test short signals: RSI cross-above rsi_upper fires entry."""
        strategy = RSIStrategy(
            rsi_period=14, rsi_lower=30, rsi_upper=70, direction="short")

        dates = pd.date_range('2024-01-01', periods=12, freq='D')
        # RSI crosses above 70 at index 4 (prev=68, curr=75)
        rsi_data = pd.Series(
            [50, 55, 60, 68, 75, 80, 78, 72, 65, 55, 45, 35], index=dates)
        prices = pd.Series(
            [100, 102, 104, 106, 108, 110, 109, 107, 105, 103, 101, 99], index=dates)

        df = pd.DataFrame({'close': prices})
        signals = strategy._generate_signals(df, rsi_data)

        buy_signal_indices = signals[signals['buy_signal']].index
        self.assertGreater(len(buy_signal_indices), 0,
                           "Short entry signal should fire on RSI cross-above rsi_upper")

        # Position should go to -1 (short) after entry
        entry_idx = signals.index.get_loc(buy_signal_indices[0])
        position_after = signals['position'].iloc[entry_idx +
                                                  1] if entry_idx + 1 < len(signals) else 0
        self.assertEqual(position_after, -1,
                         "Position should be -1 (short) after entry")

    def test_short_returns_profitable_on_price_drop(self):
        """Test that a price drop after a short entry yields positive return."""
        strategy = RSIStrategy(
            rsi_period=14, rsi_lower=30, rsi_upper=70, direction="short")

        # Build a scenario: short entry, then price drops, then cover at lower price
        dates = pd.date_range('2024-01-01', periods=20, freq='D')
        prices = [100] * 20
        rsi_vals = [50] * 20

        # RSI cross-above 70 at index 5 → short entry
        rsi_vals[4] = 68
        rsi_vals[5] = 75
        # Price then drops from 100 → 90
        prices[6] = 95
        prices[7] = 90
        # RSI cross-below 30 at index 9 → cover
        rsi_vals[8] = 35
        rsi_vals[9] = 25
        prices[9] = 85

        df = pd.DataFrame({'close': pd.Series(prices, index=dates)})
        rsi = pd.Series(rsi_vals, index=dates)

        signals = strategy._generate_signals(df, rsi)
        # Should have at least 1 buy (short entry) and 1 sell (cover)
        self.assertGreater(signals['buy_signal'].sum(),
                           0, "Should have short entry")
        self.assertGreater(signals['sell_signal'].sum(),
                           0, "Should have cover signal")

        returns = strategy._calculate_returns(df, signals, 10000.0)
        final_value = returns['portfolio_value'].iloc[-1]
        # Short profit: portfolio should be > initial cash
        self.assertGreater(final_value, 10000.0,
                           f"Short should profit on price drop; final={final_value:.2f}")

    def test_strategy_direction_defaults_to_long(self):
        """Test that strategy direction defaults to 'long'."""
        strategy = RSIStrategy(rsi_period=14, rsi_lower=30, rsi_upper=70)
        self.assertEqual(strategy.direction, "long")


if __name__ == '__main__':
    unittest.main()
