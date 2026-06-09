#!/usr/bin/env python3
"""Unit tests for positions module (current API)."""
import os
import sys
import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pandas as pd
import numpy as np

# Add app path before imports.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from positions import Position, PositionsManager  # noqa: E402
from strategy import BacktestResult  # noqa: E402


class TestPosition(unittest.TestCase):
    def test_position_creation(self):
        position = Position(
            symbol="AAPL",
            quantity=100.0,
            entry_price=150.0,
            current_price=155.0,
            current_rsi=45.0,
            entry_date=datetime(2025, 6, 14),
            alpha=0.1,
            rsi_period=14,
            rsi_lower=30,
            rsi_upper=70,
            stop_loss_price=140.0,
            take_profit_price=165.0,
        )

        self.assertEqual(position.symbol, "AAPL")
        self.assertEqual(position.quantity, 100.0)
        self.assertEqual(position.rsi_period, 14)

    def test_position_optional_fields(self):
        position = Position(
            symbol="TSLA",
            quantity=50.0,
            entry_price=800.0,
            current_price=850.0,
            current_rsi=40.0,
            entry_date=datetime(2025, 6, 14),
            alpha=0.2,
            rsi_period=14,
            rsi_lower=30,
            rsi_upper=70,
        )

        self.assertIsNone(position.stop_loss_price)
        self.assertIsNone(position.take_profit_price)
        self.assertIsNone(position.exit_reason)


class TestPositionsManager(unittest.TestCase):
    def setUp(self):
        self.cloud = Mock()
        self.data = Mock()
        self.manager = PositionsManager(self.cloud, self.data)

    def _empty_positions_df(self):
        return pd.DataFrame(columns=['symbol', 'qty', 'avg_entry_price', 'current_price', 'market_value'])

    def _empty_cloud_df(self):
        return pd.DataFrame(columns=['symbol'])

    def test_open_position_and_duplicate_guard(self):
        p = Position(
            symbol="AAPL",
            quantity=10.0,
            entry_price=100.0,
            current_price=101.0,
            current_rsi=45.0,
            entry_date=datetime.now(),
            alpha=0.1,
            rsi_period=14,
            rsi_lower=30,
            rsi_upper=70,
            closed=False,
        )

        self.manager.open_position(p)
        self.manager.open_position(p)

        self.assertEqual(len(self.manager.positions), 1)

    def test_close_position_removes_from_in_memory_list(self):
        p = Position(
            symbol="AAPL",
            quantity=10.0,
            entry_price=100.0,
            current_price=101.0,
            current_rsi=45.0,
            entry_date=datetime.now(),
            alpha=0.1,
            rsi_period=14,
            rsi_lower=30,
            rsi_upper=70,
            closed=False,
        )
        self.manager.positions = [p]
        self.cloud.get_latest_positions_df.return_value = pd.DataFrame({
            'symbol': ['AAPL'],
            'entry_price': [100.0],
            'current_price': [101.0],
            'closed': [False],
        })

        self.manager.close_position("AAPL")

        self.assertEqual(len(self.manager.positions), 0)

    def test_get_and_reconcile_positions_initializes_from_alpaca_when_cloud_empty(self):
        self.data.get_current_positions_df.return_value = pd.DataFrame({
            'symbol': ['AAPL'],
            'qty': [10.0],
            'avg_entry_price': [100.0],
            'current_price': [101.0],
            'market_value': [1010.0],
        })

        def cloud_side_effect(is_open):
            return self._empty_cloud_df() if is_open else self._empty_cloud_df()

        self.cloud.get_latest_positions_df.side_effect = cloud_side_effect

        open_positions = self.manager.get_and_reconcile_positions()

        self.assertEqual(len(open_positions), 1)
        self.assertEqual(open_positions[0].symbol, 'AAPL')
        self.assertFalse(open_positions[0].closed)

    def test_get_and_reconcile_positions_marks_cloud_only_positions_closed(self):
        self.data.get_current_positions_df.return_value = self._empty_positions_df()

        open_cloud_df = pd.DataFrame({
            'symbol': ['AAPL'],
            'shares': [10.0],
            'entry_price': [100.0],
            'current_price': [90.0],
            'position_value': [900.0],
            'current_rsi': [40.0],
            'entry_date': [datetime.now()],
            'rsi_period': [14],
            'rsi_lower': [30],
            'rsi_upper': [70],
            'alpha': [0.1],
            'stop_loss_price': [95.0],
            'take_profit_price': [110.0],
            'exit_date': [None],
            'exit_price': [pd.NA],
            'realized_return': [pd.NA],
            'closed': [False],
        })

        def cloud_side_effect(is_open):
            if is_open:
                return open_cloud_df.copy()
            return self._empty_cloud_df()

        self.cloud.get_latest_positions_df.side_effect = cloud_side_effect

        open_positions = self.manager.get_and_reconcile_positions()

        self.assertEqual(len(open_positions), 0)
        self.assertTrue(any(p.closed for p in self.manager.positions))
        closed = [p for p in self.manager.positions if p.closed]
        self.assertTrue(any(p.exit_reason == 'broker_closed' for p in closed))


    @patch('positions.globalConfig')
    def test_reconcile_alpaca_only_with_order_history(self, mock_config):
        """Test reconciliation uses order history for entry date and constrained backtest."""
        mock_config.BACKTEST_START_DATE = datetime(2026, 5, 1)
        mock_config.STOP_LOSS_PCT = 0.05
        mock_config.TAKE_PROFIT_PCT = 0.15

        self.data.get_current_positions_df.return_value = pd.DataFrame({
            'symbol': ['AAPL'],
            'qty': [10.0],
            'avg_entry_price': [150.0],
            'current_price': [155.0],
            'market_value': [1550.0],
        })

        # Return a cloud DF with a DIFFERENT symbol so AAPL triggers the
        # alpaca-only path (empty cloud would cause initialization from Alpaca).
        # Use object-dtype for date columns to avoid datetime64 precision mismatches.
        cloud_df = pd.DataFrame({
            'symbol': ['MSFT'],
            'shares': [5.0],
            'entry_price': [300.0],
            'current_price': [305.0],
            'position_value': [1525.0],
            'current_rsi': [50.0],
            'entry_date': [datetime(2026, 5, 10)],
            'rsi_period': [14],
            'rsi_lower': [30],
            'rsi_upper': [70],
            'alpha': [0.0],
            'stop_loss_price': [np.nan],
            'take_profit_price': [np.nan],
            'exit_date': [None],
            'exit_price': [np.nan],
            'realized_return': [np.nan],
            'exit_reason': [None],
            'closed': [False],
        })
        self.cloud.get_latest_positions_df.return_value = cloud_df

        # Mock order history: submitted_at = 2026-05-25
        entry_submitted = datetime(2026, 5, 25, 9, 30, 0)
        self.data.get_entry_order_for_symbol.return_value = (
            entry_submitted, 150.25
        )

        # Mock StrategyOptimizer to return a known backtest result
        with patch('strategy.StrategyOptimizer') as mock_opt_class:
            mock_opt = Mock()
            mock_opt_class.return_value = mock_opt

            backtest_result = BacktestResult(
                symbol='AAPL',
                rsi_period=14,
                rsi_lower=25,
                rsi_upper=75,
                total_return=0.05,
                buy_and_hold_return=0.03,
                alpha=0.02,
                num_trades=5,
                win_rate=0.6,
                avg_trade_duration=10,
                max_drawdown=0.02,
                sharpe_ratio=1.2,
                calmar_ratio=0.8,
                profitable=True,
                current_rsi=35.0,
                composite_score=1.5,
                direction='long',
            )
            mock_opt.optimize_symbol.return_value = backtest_result

            open_positions = self.manager.get_and_reconcile_positions()

        self.assertEqual(len(open_positions), 2)  # MSFT from cloud + AAPL from Alpaca
        aapl_pos = [p for p in open_positions if p.symbol == 'AAPL'][0]
        self.assertEqual(aapl_pos.symbol, 'AAPL')

        # entry_date should come from order history, not datetime.now()
        self.assertEqual(aapl_pos.entry_date, entry_submitted)

        # entry_price should come from order history
        self.assertEqual(aapl_pos.entry_price, 150.25)

        # RSI params should come from the backtest
        self.assertEqual(aapl_pos.rsi_period, 14)
        self.assertEqual(aapl_pos.rsi_lower, 25)
        self.assertEqual(aapl_pos.rsi_upper, 75)
        self.assertEqual(aapl_pos.alpha, 0.02)

        # Verify backtest was called with constrained window ending day before submission
        mock_opt.optimize_symbol.assert_called_once()
        call_args, call_kwargs = mock_opt.optimize_symbol.call_args
        # positional args: (symbol, start_date, end_date)
        self.assertEqual(call_args[0], 'AAPL')
        self.assertEqual(call_args[1], mock_config.BACKTEST_START_DATE)
        self.assertEqual(call_args[2], entry_submitted - timedelta(days=1))
        self.assertEqual(call_kwargs['direction'], 'long')

    @patch('positions.globalConfig')
    def test_reconcile_alpaca_only_order_history_fallback(self, mock_config):
        """Test reconciliation falls back to default behavior when no order history."""
        mock_config.BACKTEST_START_DATE = datetime(2026, 5, 1)
        mock_config.STOP_LOSS_PCT = 0.05
        mock_config.TAKE_PROFIT_PCT = 0.15

        self.data.get_current_positions_df.return_value = pd.DataFrame({
            'symbol': ['AAPL'],
            'qty': [10.0],
            'avg_entry_price': [150.0],
            'current_price': [155.0],
            'market_value': [1550.0],
        })

        # Return a cloud DF with a DIFFERENT symbol
        self.cloud.get_latest_positions_df.return_value = pd.DataFrame({
            'symbol': ['MSFT'],
            'shares': [5.0],
            'entry_price': [300.0],
            'current_price': [305.0],
            'position_value': [1525.0],
            'current_rsi': [50.0],
            'entry_date': [datetime(2026, 5, 10)],
            'rsi_period': [14],
            'rsi_lower': [30],
            'rsi_upper': [70],
            'alpha': [0.0],
            'stop_loss_price': [np.nan],
            'take_profit_price': [np.nan],
            'exit_date': [pd.NaT],
            'exit_price': [np.nan],
            'realized_return': [np.nan],
            'exit_reason': [None],
            'closed': [False],
        })

        # Mock order history returns None (no orders found)
        self.data.get_entry_order_for_symbol.return_value = None

        # Mock StrategyOptimizer
        with patch('strategy.StrategyOptimizer') as mock_opt_class:
            mock_opt = Mock()
            mock_opt_class.return_value = mock_opt

            backtest_result = BacktestResult(
                symbol='AAPL',
                rsi_period=7,
                rsi_lower=20,
                rsi_upper=80,
                total_return=0.10,
                buy_and_hold_return=0.02,
                alpha=0.08,
                num_trades=8,
                win_rate=0.75,
                avg_trade_duration=8,
                max_drawdown=0.03,
                sharpe_ratio=1.5,
                profitable=True,
                current_rsi=42.0,
                direction='long',
            )
            mock_opt.optimize_symbol.return_value = backtest_result

            open_positions = self.manager.get_and_reconcile_positions()

        self.assertEqual(len(open_positions), 2)  # MSFT from cloud + AAPL from Alpaca
        aapl_pos = [p for p in open_positions if p.symbol == 'AAPL'][0]
        self.assertEqual(aapl_pos.symbol, 'AAPL')
        self.assertEqual(aapl_pos.rsi_period, 7)
        self.assertEqual(aapl_pos.rsi_lower, 20)
        self.assertEqual(aapl_pos.rsi_upper, 80)

        # Verify backtest was called with the full range (default behavior)
        mock_opt.optimize_symbol.assert_called_once()
        call_args, call_kwargs = mock_opt.optimize_symbol.call_args
        self.assertEqual(call_args[0], 'AAPL')
        self.assertEqual(call_args[1], mock_config.BACKTEST_START_DATE)
        # end_date should be near now (datetime.now() - 20 min), not constrained
        self.assertIsNotNone(call_args[2])

    @patch('positions.globalConfig')
    def test_reconcile_alpaca_only_backtest_skipped_if_entry_too_old(self, mock_config):
        """Test backtest is skipped when entry date makes backtest window empty."""
        mock_config.BACKTEST_START_DATE = datetime(2026, 5, 20)
        mock_config.STOP_LOSS_PCT = 0.05
        mock_config.TAKE_PROFIT_PCT = 0.15

        self.data.get_current_positions_df.return_value = pd.DataFrame({
            'symbol': ['AAPL'],
            'qty': [10.0],
            'avg_entry_price': [150.0],
            'current_price': [155.0],
            'market_value': [1550.0],
        })

        # Return a cloud DF with a DIFFERENT symbol
        self.cloud.get_latest_positions_df.return_value = pd.DataFrame({
            'symbol': ['MSFT'],
            'shares': [5.0],
            'entry_price': [300.0],
            'current_price': [305.0],
            'position_value': [1525.0],
            'current_rsi': [50.0],
            'entry_date': [datetime(2026, 5, 10)],
            'rsi_period': [14],
            'rsi_lower': [30],
            'rsi_upper': [70],
            'alpha': [0.0],
            'stop_loss_price': [np.nan],
            'take_profit_price': [np.nan],
            'exit_date': [pd.NaT],
            'exit_price': [np.nan],
            'realized_return': [np.nan],
            'exit_reason': [None],
            'closed': [False],
        })

        # Entry submitted_at BEFORE BACKTEST_START_DATE
        # end_date = submitted_at - 1 day, which is before start
        early_date = datetime(2026, 5, 10)
        self.data.get_entry_order_for_symbol.return_value = (early_date, 145.0)

        with patch('strategy.StrategyOptimizer') as mock_opt_class:
            mock_opt = Mock()
            mock_opt_class.return_value = mock_opt

            open_positions = self.manager.get_and_reconcile_positions()

        self.assertEqual(len(open_positions), 2)  # MSFT from cloud + AAPL from Alpaca
        aapl_pos = [p for p in open_positions if p.symbol == 'AAPL'][0]
        self.assertEqual(aapl_pos.symbol, 'AAPL')

        # Should use default RSI parameters since backtest was skipped
        self.assertEqual(aapl_pos.rsi_period, 14)
        self.assertEqual(aapl_pos.rsi_lower, 30)
        self.assertEqual(aapl_pos.rsi_upper, 70)
        self.assertEqual(aapl_pos.alpha, 0.0)

        # entry_date should still come from order history
        self.assertEqual(aapl_pos.entry_date, early_date)

        # entry_price should come from order history
        self.assertEqual(aapl_pos.entry_price, 145.0)

        # Backtest should NOT have been called
        mock_opt.optimize_symbol.assert_not_called()


if __name__ == '__main__':
    unittest.main()
