#!/usr/bin/env python3
"""
Unit tests for the StrategyOptimizer module.
"""
import os
import sys
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from strategy import BacktestResult  # noqa: E402
from optimizer import StrategyOptimizer  # noqa: E402


class TestStrategyOptimizer(unittest.TestCase):
    """Test cases for the StrategyOptimizer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.optimizer = StrategyOptimizer()

    def test_optimizer_init(self):
        """Test StrategyOptimizer initialization."""
        self.assertIsNotNone(self.optimizer.rsi_periods)
        self.assertIsNotNone(self.optimizer.rsi_lowers)
        self.assertIsNotNone(self.optimizer.rsi_uppers)
        self.assertIsInstance(
            self.optimizer.last_consolidated_trades_df, pd.DataFrame)

    def test_filter_results_filters_unprofitable(self):
        """Test that filter_results removes unprofitable results."""
        results = [
            BacktestResult(
                symbol="AAPL", rsi_period=14, rsi_lower=30, rsi_upper=70,
                total_return=0.15, buy_and_hold_return=0.10, alpha=0.05,
                num_trades=5, win_rate=0.6, avg_trade_duration=10.5,
                max_drawdown=0.08, sharpe_ratio=1.2, calmar_ratio=2.0,
                profitable=True,
            ),
            BacktestResult(
                symbol="TSLA", rsi_period=14, rsi_lower=30, rsi_upper=70,
                total_return=-0.10, buy_and_hold_return=0.05, alpha=-0.15,
                num_trades=3, win_rate=0.33, avg_trade_duration=5.0,
                max_drawdown=0.20, sharpe_ratio=-0.5, calmar_ratio=-0.5,
                profitable=False,
            ),
        ]
        results[0].composite_score = 3.0
        results[1].composite_score = -2.0

        filtered = self.optimizer.filter_results(results)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].symbol, "AAPL")

    def test_filter_results_sorts_by_composite_score(self):
        """Test that filter_results sorts by composite_score descending."""
        r1 = BacktestResult(
            symbol="AAPL", rsi_period=14, rsi_lower=30, rsi_upper=70,
            total_return=0.15, buy_and_hold_return=0.10, alpha=0.05,
            num_trades=5, win_rate=0.6, avg_trade_duration=10.5,
            max_drawdown=0.08, sharpe_ratio=1.2, calmar_ratio=2.0,
            profitable=True,
        )
        r1.composite_score = 2.0

        r2 = BacktestResult(
            symbol="MSFT", rsi_period=14, rsi_lower=30, rsi_upper=70,
            total_return=0.20, buy_and_hold_return=0.10, alpha=0.10,
            num_trades=8, win_rate=0.75, avg_trade_duration=12.0,
            max_drawdown=0.06, sharpe_ratio=1.8, calmar_ratio=3.0,
            profitable=True,
        )
        r2.composite_score = 5.0

        filtered = self.optimizer.filter_results([r1, r2])
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0].symbol, "MSFT")  # Higher score first
        self.assertEqual(filtered[1].symbol, "AAPL")

    def test_filter_results_requires_min_win_rate(self):
        """Test that filter_results rejects results with win_rate <= 0.3."""
        result = BacktestResult(
            symbol="AAPL", rsi_period=14, rsi_lower=30, rsi_upper=70,
            total_return=0.15, buy_and_hold_return=0.10, alpha=0.05,
            num_trades=10, win_rate=0.2, avg_trade_duration=10.5,
            max_drawdown=0.08, sharpe_ratio=1.2, calmar_ratio=2.0,
            profitable=True,
        )
        result.composite_score = 3.0

        filtered = self.optimizer.filter_results([result])
        self.assertEqual(len(filtered), 0)

    def test_filter_results_requires_trades(self):
        """Test that filter_results rejects results with zero trades."""
        result = BacktestResult(
            symbol="AAPL", rsi_period=14, rsi_lower=30, rsi_upper=70,
            total_return=0.0, buy_and_hold_return=0.10, alpha=-0.10,
            num_trades=0, win_rate=0.0, avg_trade_duration=0.0,
            max_drawdown=0.0, sharpe_ratio=0.0, calmar_ratio=0.0,
            profitable=True,
        )
        result.composite_score = 0.0

        filtered = self.optimizer.filter_results([result])
        self.assertEqual(len(filtered), 0)

    def test_optimize_symbol_no_data(self):
        """Test optimize_symbol returns None when symbol has no data."""
        with patch('optimizer.data_provider') as mock_dp:
            mock_dp.get_single_stock_bars.return_value = pd.DataFrame()
            result = self.optimizer.optimize_symbol(
                "INVALID", datetime(2026, 1, 1), datetime(2026, 6, 1), "long"
            )
            self.assertIsNone(result)

    def test_build_consolidated_trades_empty(self):
        """Test build_consolidated_trades with no results."""
        df = self.optimizer.build_consolidated_trades([])
        self.assertTrue(df.empty)

    def test_build_consolidated_trades_with_data(self):
        """Test build_consolidated_trades with trade data."""
        trade_details = [{
            'entry_date': pd.Timestamp('2026-01-15'),
            'exit_date': pd.Timestamp('2026-01-20'),
            'entry_price': 150.0,
            'exit_price': 155.0,
            'return': 0.0333,
            'duration': 5,
            'exit_reason': 'rsi_cross',
            'direction': 'long',
        }]
        result = BacktestResult(
            symbol="AAPL", rsi_period=14, rsi_lower=30, rsi_upper=70,
            total_return=0.15, buy_and_hold_return=0.10, alpha=0.05,
            num_trades=1, win_rate=1.0, avg_trade_duration=5.0,
            max_drawdown=0.03, sharpe_ratio=1.5, calmar_ratio=2.0,
            profitable=True, trade_details=trade_details,
        )

        df = self.optimizer.build_consolidated_trades([result])
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['symbol'], "AAPL")
        self.assertEqual(df.iloc[0]['exit_reason'], "rsi_cross")
