#!/usr/bin/env python3
"""
Unit tests for the strategy module.
"""
import os
import sys
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
from strategy import BacktestResult, RSIStrategy, StrategyBacktester

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))


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

    @patch('strategy.config')
    def test_rsi_strategy_init_with_config_max_hold_days(self, mock_config):
        """Test RSIStrategy initialization using globalConfig for max_hold_days."""
        mock_config.MAX_HOLD_DAYS = 45

        strategy = RSIStrategy(rsi_period=21, rsi_lower=25, rsi_upper=75)

        self.assertEqual(strategy.max_hold_days, 45)

    def test_calculate_rsi(self):
        """Test RSI calculation method."""
        # Create sample price data
        prices = pd.Series([100, 105, 103, 108, 110, 107,
                           112, 115, 113, 118, 120, 117, 122, 125, 123, 128])

        rsi = self.strategy._calculate_rsi(prices)

        self.assertIsInstance(rsi, pd.Series)
        self.assertEqual(len(rsi), len(prices))
        # Check that RSI values are within valid range (0-100)
        valid_rsi = rsi.dropna()
        self.assertTrue(all(0 <= val <= 100 for val in valid_rsi))

    def test_generate_signals(self):
        """Test signal generation."""
        # Create sample RSI data
        rsi_data = pd.Series(
            [70, 75, 80, 60, 25, 20, 35, 45, 75, 80, 25, 15, 40])
        prices = pd.Series([100, 105, 110, 108, 95, 90,
                           98, 102, 110, 115, 95, 85, 95])

        df = pd.DataFrame({'close': prices, 'rsi': rsi_data})

        signals = self.strategy._generate_signals(df)

        self.assertIn('buy_signal', signals.columns)
        self.assertIn('sell_signal', signals.columns)

        # Check that we get buy signals when RSI is low
        buy_signals = signals[signals['buy_signal'] == True]
        self.assertTrue(len(buy_signals) > 0)

    def test_calculate_returns(self):
        """Test returns calculation."""
        # Create sample data with signals
        data = {
            'close': [100, 105, 110, 108, 95, 90, 98, 102, 110, 115],
            'buy_signal': [False, False, False, False, True, False, False, False, False, False],
            'sell_signal': [False, False, False, False, False, False, False, True, False, False]
        }
        df = pd.DataFrame(data)

        returns = self.strategy._calculate_returns(df)

        self.assertIn('strategy_return', returns.columns)
        self.assertIn('buy_and_hold_return', returns.columns)

    def test_backtest_with_insufficient_data(self):
        """Test backtest with insufficient data."""
        # Create insufficient data (less than RSI period)
        prices = pd.Series([100, 105, 103])
        df = pd.DataFrame({'close': prices})

        result = self.strategy.backtest("TEST", df)

        self.assertIsNone(result)

    def test_backtest_with_valid_data(self):
        """Test backtest with valid data."""
        # Create sufficient sample data
        np.random.seed(42)  # For reproducible results
        prices = pd.Series(100 + np.cumsum(np.random.randn(100) * 0.02))
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        df = pd.DataFrame({'close': prices}, index=dates)

        result = self.strategy.backtest("TEST", df)

        self.assertIsInstance(result, BacktestResult)
        self.assertEqual(result.symbol, "TEST")
        self.assertEqual(result.rsi_period, 14)
        self.assertEqual(result.rsi_lower, 30)
        self.assertEqual(result.rsi_upper, 70)


class TestStrategyBacktester(unittest.TestCase):
    """Test cases for the StrategyBacktester class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_data_provider = Mock()

    @patch('strategy.data_provider')
    def test_strategy_backtester_init(self, mock_data_provider):
        """Test StrategyBacktester initialization."""
        backtester = StrategyBacktester()

        self.assertIsNotNone(backtester)

    @patch('strategy.data_provider')
    def test_run_single_backtest(self, mock_data_provider):
        """Test running a single backtest."""
        # Mock data provider
        sample_data = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(50) * 0.02)
        }, index=pd.date_range('2024-01-01', periods=50, freq='D'))

        mock_data_provider.get_historical_data.return_value = sample_data

        backtester = StrategyBacktester()

        result = backtester.run_single_backtest("AAPL", 14, 30, 70)

        self.assertIsInstance(result, BacktestResult)
        self.assertEqual(result.symbol, "AAPL")

    @patch('strategy.data_provider')
    def test_run_single_backtest_no_data(self, mock_data_provider):
        """Test running backtest when no data is available."""
        mock_data_provider.get_historical_data.return_value = None

        backtester = StrategyBacktester()

        result = backtester.run_single_backtest("INVALID", 14, 30, 70)

        self.assertIsNone(result)

    @patch('strategy.data_provider')
    def test_run_parallel_backtests(self, mock_data_provider):
        """Test running parallel backtests."""
        # Mock data provider
        sample_data = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(100) * 0.02)
        }, index=pd.date_range('2024-01-01', periods=100, freq='D'))

        mock_data_provider.get_historical_data.return_value = sample_data

        backtester = StrategyBacktester()

        symbols = ["AAPL", "TSLA"]
        rsi_configs = [(14, 30, 70), (21, 25, 75)]

        results = backtester.run_parallel_backtests(symbols, rsi_configs)

        self.assertIsInstance(results, list)
        self.assertTrue(len(results) > 0)

        # Check that all results are BacktestResult objects
        for result in results:
            if result is not None:
                self.assertIsInstance(result, BacktestResult)

    @patch('strategy.data_provider')
    def test_optimize_parameters(self, mock_data_provider):
        """Test parameter optimization."""
        # Mock data provider
        sample_data = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(100) * 0.02)
        }, index=pd.date_range('2024-01-01', periods=100, freq='D'))

        mock_data_provider.get_historical_data.return_value = sample_data

        backtester = StrategyBacktester()

        best_result = backtester.optimize_parameters("AAPL")

        if best_result is not None:
            self.assertIsInstance(best_result, BacktestResult)
            self.assertEqual(best_result.symbol, "AAPL")

    @patch('strategy.data_provider')
    def test_get_current_rsi(self, mock_data_provider):
        """Test getting current RSI value."""
        # Mock data provider
        sample_data = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(30) * 0.02)
        }, index=pd.date_range('2024-01-01', periods=30, freq='D'))

        mock_data_provider.get_recent_data.return_value = sample_data

        backtester = StrategyBacktester()

        current_rsi = backtester.get_current_rsi("AAPL", 14)

        if current_rsi is not None:
            self.assertIsInstance(current_rsi, (int, float))
            self.assertTrue(0 <= current_rsi <= 100)


if __name__ == '__main__':
    unittest.main()
