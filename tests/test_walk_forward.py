#!/usr/bin/env python3
"""
Unit tests for the WalkForwardValidator module.
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
from walk_forward import (  # noqa: E402
    WalkForwardResult,
    WalkForwardValidator,
    WalkForwardWindow,
)


class TestWalkForwardWindow(unittest.TestCase):
    """Test cases for WalkForwardWindow dataclass."""

    def test_window_creation(self):
        """Test creating a WalkForwardWindow."""
        win = WalkForwardWindow(
            window_index=0,
            is_start=datetime(2026, 1, 1),
            is_end=datetime(2026, 4, 1),
            oos_start=datetime(2026, 4, 1),
            oos_end=datetime(2026, 6, 1),
            best_period=14,
            best_lower=30,
            best_upper=70,
            is_total_return=0.10,
            is_sharpe_ratio=1.2,
            is_num_trades=5,
            oos_total_return=0.05,
            oos_sharpe_ratio=0.8,
            oos_max_drawdown=0.04,
            oos_win_rate=0.6,
            oos_num_trades=3,
            oos_profitable=True,
            is_optimized=True,
            oos_validated=True,
        )
        self.assertEqual(win.window_index, 0)
        self.assertTrue(win.is_optimized)
        self.assertTrue(win.oos_validated)
        self.assertTrue(win.oos_profitable)

    def test_window_defaults(self):
        """Test WalkForwardWindow default values."""
        win = WalkForwardWindow(
            window_index=1,
            is_start=datetime(2026, 1, 1),
            is_end=datetime(2026, 4, 1),
            oos_start=datetime(2026, 4, 1),
            oos_end=datetime(2026, 6, 1),
            best_period=14,
            best_lower=30,
            best_upper=70,
            is_total_return=0.0,
            is_sharpe_ratio=0.0,
            is_num_trades=0,
        )
        self.assertFalse(win.is_optimized)
        self.assertFalse(win.oos_validated)
        self.assertFalse(win.oos_profitable)
        self.assertIsNone(win.error)


class TestWalkForwardResult(unittest.TestCase):
    """Test cases for WalkForwardResult dataclass."""

    def test_result_creation(self):
        """Test creating a WalkForwardResult."""
        result = WalkForwardResult(symbol="AAPL", direction="long")
        self.assertEqual(result.symbol, "AAPL")
        self.assertEqual(result.direction, "long")
        self.assertEqual(result.num_windows, 0)
        self.assertEqual(result.num_profitable_oos_windows, 0)

    def test_num_profitable_oos_windows(self):
        """Test counting profitable OOS windows."""
        windows = [
            WalkForwardWindow(
                window_index=i,
                is_start=datetime(2026, 1, 1),
                is_end=datetime(2026, 4, 1),
                oos_start=datetime(2026, 4, 1),
                oos_end=datetime(2026, 6, 1),
                best_period=14, best_lower=30, best_upper=70,
                is_total_return=0.10, is_sharpe_ratio=1.0, is_num_trades=5,
                oos_total_return=0.05, oos_sharpe_ratio=0.5,
                oos_max_drawdown=0.03, oos_win_rate=0.6, oos_num_trades=3,
                oos_profitable=(i % 2 == 0),  # Even = profitable
                is_optimized=True, oos_validated=True,
            )
            for i in range(4)
        ]
        result = WalkForwardResult(
            symbol="AAPL", direction="long", windows=windows)
        self.assertEqual(result.num_windows, 4)
        self.assertEqual(result.num_profitable_oos_windows,
                         2)  # windows 0 and 2

    def test_to_backtest_result(self):
        """Test converting WalkForwardResult to BacktestResult."""
        wf_result = WalkForwardResult(
            symbol="AAPL",
            direction="long",
            oos_total_return=0.08,
            oos_sharpe_ratio=1.5,
            oos_max_drawdown=0.05,
            oos_win_rate=0.65,
            oos_num_trades=12,
            oos_calmar_ratio=2.0,
            best_rsi_period=14,
            best_rsi_lower=30,
            best_rsi_upper=70,
            composite_score=3.5,
            profitable=True,
            alpha=0.08,
        )

        bt_result = wf_result.to_backtest_result()
        self.assertEqual(bt_result.symbol, "AAPL")
        self.assertEqual(bt_result.rsi_period, 14)
        self.assertEqual(bt_result.rsi_lower, 30)
        self.assertEqual(bt_result.rsi_upper, 70)
        self.assertAlmostEqual(bt_result.total_return, 0.08)
        self.assertAlmostEqual(bt_result.sharpe_ratio, 1.5)
        self.assertAlmostEqual(bt_result.composite_score, 3.5)
        self.assertTrue(bt_result.profitable)
        self.assertEqual(bt_result.direction, "long")
        self.assertAlmostEqual(bt_result.alpha, 0.08)

    def test_to_backtest_result_defaults(self):
        """Test to_backtest_result with None parameters uses defaults."""
        wf_result = WalkForwardResult(symbol="TSLA", direction="short")
        bt_result = wf_result.to_backtest_result()
        self.assertEqual(bt_result.symbol, "TSLA")
        self.assertEqual(bt_result.rsi_period, 14)  # default
        self.assertEqual(bt_result.rsi_lower, 30)  # default
        self.assertEqual(bt_result.rsi_upper, 70)  # default
        self.assertEqual(bt_result.direction, "short")


class TestWalkForwardValidator(unittest.TestCase):
    """Test cases for WalkForwardValidator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_optimizer = Mock()
        self.validator = WalkForwardValidator(self.mock_optimizer)

    def test_compute_window_boundaries_basic(self):
        """Test window boundary computation with standard parameters."""
        with patch('walk_forward.globalConfig') as mock_config:
            mock_config.WF_IS_MONTHS = 6
            mock_config.WF_OOS_MONTHS = 2
            mock_config.WF_STEP_MONTHS = 2
            mock_config.WF_MIN_WINDOWS = 3

            start = datetime(2025, 1, 1)
            end = datetime(2026, 1, 1)

            windows = self.validator._compute_window_boundaries(start, end)

            # With 12 months: (0-6, 6-8), (2-8, 8-10), (4-10, 10-12)
            self.assertEqual(len(windows), 3)

            # Check first window
            is_start, is_end, oos_start, oos_end = windows[0]
            self.assertEqual(is_start, start)
            self.assertAlmostEqual((is_end - is_start).days, 180, delta=3)
            self.assertEqual(oos_start, is_end)
            self.assertAlmostEqual((oos_end - oos_start).days, 60, delta=3)

            # Windows should be sequential
            self.assertEqual(windows[1][0], start + timedelta(days=60))

    def test_compute_window_boundaries_insufficient_data(self):
        """Test window boundary computation with insufficient data."""
        with patch('walk_forward.globalConfig') as mock_config:
            mock_config.WF_IS_MONTHS = 6
            mock_config.WF_OOS_MONTHS = 2
            mock_config.WF_STEP_MONTHS = 2
            mock_config.WF_MIN_WINDOWS = 3

            start = datetime(2025, 1, 1)
            end = datetime(2025, 4, 1)  # Only 3 months

            windows = self.validator._compute_window_boundaries(start, end)
            self.assertEqual(len(windows), 0)

    def test_compute_window_boundaries_stops_at_oos_end(self):
        """Test that windows stop when OOS end would exceed available data."""
        with patch('walk_forward.globalConfig') as mock_config:
            mock_config.WF_IS_MONTHS = 6
            mock_config.WF_OOS_MONTHS = 2
            mock_config.WF_STEP_MONTHS = 2
            mock_config.WF_MIN_WINDOWS = 1

            start = datetime(2025, 1, 1)
            end = datetime(2025, 9, 1)  # 8 months

            windows = self.validator._compute_window_boundaries(start, end)
            # 8 months allows: (0-6, 6-8) = 1 window; (2-8, 8-10) would exceed end
            self.assertEqual(len(windows), 1)

    @patch('walk_forward.globalConfig')
    def test_validate_symbol_insufficient_windows(self, mock_config):
        """Test that validate_symbol returns None when too few windows."""
        mock_config.WF_MIN_WINDOWS = 3
        mock_config.WF_IS_MONTHS = 6
        mock_config.WF_OOS_MONTHS = 2
        mock_config.WF_STEP_MONTHS = 2

        start = datetime(2025, 1, 1)
        end = datetime(2025, 6, 1)  # Only 5 months — not enough

        result = self.validator.validate_symbol("AAPL", start, end, "long")
        self.assertIsNone(result)

    @patch('walk_forward.globalConfig')
    def test_validate_symbol_all_fail_is(self, mock_config):
        """Test validate_symbol when all IS optimizations fail."""
        mock_config.WF_MIN_WINDOWS = 1
        mock_config.WF_IS_MONTHS = 6
        mock_config.WF_OOS_MONTHS = 2
        mock_config.WF_STEP_MONTHS = 2
        mock_config.BACKTEST_INIT_CASH = 10000

        # Mock optimizer to always return None (IS failure)
        self.mock_optimizer.optimize_symbol.return_value = None

        start = datetime(2025, 1, 1)
        end = datetime(2026, 1, 1)

        result = self.validator.validate_symbol("AAPL", start, end, "long")
        self.assertIsNotNone(result)
        self.assertEqual(len(result.windows), 3)
        self.assertFalse(any(w.is_optimized for w in result.windows))
        self.assertFalse(result.profitable)

    @patch('walk_forward.globalConfig')
    @patch('data_provider.data_provider')
    def test_validate_symbol_successful(self, mock_dp, mock_config):
        """Test full walk-forward with successful IS and OOS."""
        mock_config.WF_MIN_WINDOWS = 1
        mock_config.WF_IS_MONTHS = 6
        mock_config.WF_OOS_MONTHS = 2
        mock_config.WF_STEP_MONTHS = 2
        mock_config.BACKTEST_INIT_CASH = 10000

        # Mock IS optimizer: returns profitable result
        is_result = BacktestResult(
            symbol="AAPL", rsi_period=14, rsi_lower=30, rsi_upper=70,
            total_return=0.15, buy_and_hold_return=0.10, alpha=0.05,
            num_trades=8, win_rate=0.75, avg_trade_duration=10.0,
            max_drawdown=0.06, sharpe_ratio=1.5, calmar_ratio=2.5,
            profitable=True,
        )
        self.mock_optimizer.optimize_symbol.return_value = is_result

        # Mock OOS backtest data
        mock_data = pd.DataFrame({
            'close': np.linspace(100, 110, 30),
            'datetime': pd.date_range('2025-07-01', periods=30),
        })
        mock_data.set_index('datetime', inplace=True)
        mock_dp.get_single_stock_bars.return_value = mock_data

        start = datetime(2025, 1, 1)
        end = datetime(2026, 1, 1)

        result = self.validator.validate_symbol("AAPL", start, end, "long")
        self.assertIsNotNone(result)
        self.assertEqual(result.symbol, "AAPL")
        self.assertEqual(result.direction, "long")
        self.assertEqual(len(result.windows), 3)
        # All windows should have been IS-optimized and OOS-validated
        self.assertTrue(all(w.is_optimized for w in result.windows))
        self.assertTrue(all(w.oos_validated for w in result.windows))
        self.assertEqual(result.best_rsi_period, 14)
        self.assertEqual(result.best_rsi_lower, 30)
        self.assertEqual(result.best_rsi_upper, 70)

    @patch('walk_forward.globalConfig')
    def test_compute_wf_cross_symbol_zscores(self, mock_config):
        """Test cross-symbol Z-score computation."""
        r1 = WalkForwardResult(
            symbol="AAPL", direction="long",
            oos_total_return=0.10, oos_sharpe_ratio=1.5,
            oos_calmar_ratio=2.0,
        )
        r2 = WalkForwardResult(
            symbol="MSFT", direction="long",
            oos_total_return=0.05, oos_sharpe_ratio=0.8,
            oos_calmar_ratio=1.0,
        )
        r3 = WalkForwardResult(
            symbol="TSLA", direction="long",
            oos_total_return=0.15, oos_sharpe_ratio=2.0,
            oos_calmar_ratio=3.0,
        )

        results = [r1, r2, r3]
        WalkForwardValidator._compute_wf_cross_symbol_zscores(results)

        # All should have non-zero composite scores
        for r in results:
            self.assertNotEqual(r.composite_score, 0.0)

        # TSLA has best metrics, should have highest Z-score
        self.assertGreater(r3.composite_score, r2.composite_score)

    def test_compute_wf_cross_symbol_zscores_single_result(self):
        """Test Z-score with a single result (no pool for normalization)."""
        r1 = WalkForwardResult(
            symbol="AAPL", direction="long",
            oos_total_return=0.10, oos_sharpe_ratio=1.5,
            oos_calmar_ratio=2.0,
        )
        WalkForwardValidator._compute_wf_cross_symbol_zscores([r1])
        self.assertEqual(r1.composite_score, 0.0)

    def test_param_stability_computation(self):
        """Test parameter stability calculation."""
        windows = []
        params_sequence = [
            (14, 30, 70),  # window 0
            (14, 30, 70),  # window 1 — same as window 0
            (14, 25, 75),  # window 2 — different
            (14, 30, 70),  # window 3 — same as window 0
        ]
        for i, (period, lower, upper) in enumerate(params_sequence):
            windows.append(WalkForwardWindow(
                window_index=i,
                is_start=datetime(2025, 1, 1),
                is_end=datetime(2025, 7, 1),
                oos_start=datetime(2025, 7, 1),
                oos_end=datetime(2025, 9, 1),
                best_period=period, best_lower=lower, best_upper=upper,
                is_total_return=0.10, is_sharpe_ratio=1.0, is_num_trades=5,
                is_optimized=True, oos_validated=True,
                oos_total_return=0.05, oos_sharpe_ratio=0.5,
                oos_max_drawdown=0.04, oos_win_rate=0.6, oos_num_trades=3,
                oos_profitable=True,
            ))

        result = WalkForwardResult(
            symbol="AAPL", direction="long", windows=windows,
        )
        # Manually trigger aggregation to compute param_stability
        # The _aggregate_windows method computes it; we call it indirectly
        # via the validator's internal method
        validator = WalkForwardValidator(Mock())
        aggregated = validator._aggregate_windows("AAPL", "long", windows)
        # (14, 30, 70) appears in 3 out of 4 windows = 0.75
        self.assertAlmostEqual(aggregated.param_stability, 0.75)
