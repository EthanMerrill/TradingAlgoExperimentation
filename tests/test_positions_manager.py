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
        # Default: no order history (empty DataFrame) — individual tests
        # that need fills can override this.
        self.data.get_filled_orders_for_symbol.return_value = pd.DataFrame()

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

    def test_close_position_marks_closed_in_place(self):
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
            stop_loss_price=95.0,
            take_profit_price=110.0,
            closed=False,
        )
        self.manager.positions = [p]
        self.cloud.get_latest_positions_df.return_value = pd.DataFrame({
            'symbol': ['AAPL'],
            'entry_price': [100.0],
            'current_price': [101.0],
            'stop_loss_price': [95.0],
            'take_profit_price': [110.0],
            'closed': [False],
        })

        self.manager.close_position("AAPL")

        # Position stays in list but is now closed
        self.assertEqual(len(self.manager.positions), 1)
        self.assertTrue(self.manager.positions[0].closed)
        self.assertIsNotNone(self.manager.positions[0].exit_price)
        self.assertIsNotNone(self.manager.positions[0].realized_return)

    def test_close_short_position_correct_realized_return(self):
        """Short that lost money: entry=100, covered at 105 → -5% return."""
        p = Position(
            symbol="PDD",
            quantity=-10.0,  # negative = short
            entry_price=100.0,
            current_price=110.0,
            current_rsi=70.0,
            entry_date=datetime.now(),
            alpha=0.1,
            rsi_period=14,
            rsi_lower=30,
            rsi_upper=70,
            stop_loss_price=105.0,
            take_profit_price=90.0,
            closed=False,
        )
        self.manager.positions = [p]
        self.cloud.get_latest_positions_df.return_value = pd.DataFrame({
            'symbol': ['PDD'],
            'entry_price': [100.0],
            'current_price': [110.0],
            'shares': [-10.0],
            'stop_loss_price': [105.0],
            'take_profit_price': [90.0],
            'closed': [False],
        })

        self.manager.close_position("PDD")

        self.assertEqual(len(self.manager.positions), 1)
        self.assertTrue(self.manager.positions[0].closed)
        # OCO fallback: 110 is closer to 105 (stop) than 90 (take), so exit=105
        self.assertEqual(self.manager.positions[0].exit_price, 105.0)
        # Short: (100 - 105) / 100 = -0.05
        self.assertAlmostEqual(
            self.manager.positions[0].realized_return, -0.05, places=4)

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

        def _cloud_df_side_effect(is_open):
            if is_open:
                return cloud_df.copy()
            return self._empty_cloud_df()
        self.cloud.get_latest_positions_df.side_effect = _cloud_df_side_effect

        # Mock order history: submitted_at = 2026-05-25
        entry_submitted = datetime(2026, 5, 25, 9, 30, 0)
        self.data.get_entry_order_for_symbol.return_value = (
            entry_submitted, 150.25
        )

        # Mock StrategyOptimizer to return a known backtest result
        with patch('optimizer.StrategyOptimizer') as mock_opt_class:
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

        # Only AAPL should be open; MSFT is cloud-only (not in Alpaca)
        # and should have been marked broker_closed.
        self.assertEqual(len(open_positions), 1)
        aapl_pos = [p for p in open_positions if p.symbol == 'AAPL'][0]
        self.assertEqual(aapl_pos.symbol, 'AAPL')

        # MSFT should be in self.manager.positions as closed (broker_closed)
        msft_positions = [
            p for p in self.manager.positions if p.symbol == 'MSFT']
        self.assertEqual(len(msft_positions), 1)
        self.assertTrue(msft_positions[0].closed)
        self.assertEqual(msft_positions[0].exit_reason, 'broker_closed')

        # entry_date should come from order history, not datetime.now()
        self.assertEqual(aapl_pos.entry_date, entry_submitted)

        # entry_price should come from Alpaca (source of truth)
        self.assertEqual(aapl_pos.entry_price, 150.0)

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
        fallback_cloud_df = pd.DataFrame({
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

        def _fallback_cloud_df_side_effect(is_open):
            if is_open:
                return fallback_cloud_df.copy()
            return self._empty_cloud_df()
        self.cloud.get_latest_positions_df.side_effect = _fallback_cloud_df_side_effect

        # Mock order history returns None (no orders found)
        self.data.get_entry_order_for_symbol.return_value = None

        # Mock StrategyOptimizer
        with patch('optimizer.StrategyOptimizer') as mock_opt_class:
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

        # Only AAPL should be open; MSFT is cloud-only (not in Alpaca)
        # and should have been marked broker_closed.
        self.assertEqual(len(open_positions), 1)
        aapl_pos = [p for p in open_positions if p.symbol == 'AAPL'][0]
        self.assertEqual(aapl_pos.symbol, 'AAPL')
        self.assertEqual(aapl_pos.rsi_period, 7)
        self.assertEqual(aapl_pos.rsi_lower, 20)
        self.assertEqual(aapl_pos.rsi_upper, 80)

        # MSFT should be in self.manager.positions as closed (broker_closed)
        msft_positions = [
            p for p in self.manager.positions if p.symbol == 'MSFT']
        self.assertEqual(len(msft_positions), 1)
        self.assertTrue(msft_positions[0].closed)
        self.assertEqual(msft_positions[0].exit_reason, 'broker_closed')

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
        stale_cloud_df = pd.DataFrame({
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

        def _stale_cloud_df_side_effect(is_open):
            if is_open:
                return stale_cloud_df.copy()
            return self._empty_cloud_df()
        self.cloud.get_latest_positions_df.side_effect = _stale_cloud_df_side_effect

        # Entry submitted_at BEFORE BACKTEST_START_DATE
        # end_date = submitted_at - 1 day, which is before start
        early_date = datetime(2026, 5, 10)
        self.data.get_entry_order_for_symbol.return_value = (early_date, 145.0)

        with patch('optimizer.StrategyOptimizer') as mock_opt_class:
            mock_opt = Mock()
            mock_opt_class.return_value = mock_opt

            open_positions = self.manager.get_and_reconcile_positions()

        # Only AAPL should be open; MSFT is cloud-only (not in Alpaca)
        # and should have been marked broker_closed.
        self.assertEqual(len(open_positions), 1)
        aapl_pos = [p for p in open_positions if p.symbol == 'AAPL'][0]
        self.assertEqual(aapl_pos.symbol, 'AAPL')

        # MSFT should be in self.manager.positions as closed (broker_closed)
        msft_positions = [
            p for p in self.manager.positions if p.symbol == 'MSFT']
        self.assertEqual(len(msft_positions), 1)
        self.assertTrue(msft_positions[0].closed)
        self.assertEqual(msft_positions[0].exit_reason, 'broker_closed')

        # Should use default RSI parameters since backtest was skipped
        self.assertEqual(aapl_pos.rsi_period, 14)
        self.assertEqual(aapl_pos.rsi_lower, 30)
        self.assertEqual(aapl_pos.rsi_upper, 70)
        self.assertEqual(aapl_pos.alpha, 0.0)

        # entry_date should still come from order history
        self.assertEqual(aapl_pos.entry_date, early_date)

        # entry_price should come from Alpaca (source of truth), not order history
        self.assertEqual(aapl_pos.entry_price, 150.0)

        # Backtest should NOT have been called
        mock_opt.optimize_symbol.assert_not_called()

    @patch('positions.globalConfig')
    def test_reconcile_alpaca_only_short_position(self, mock_config):
        """Alpaca-only SHORT: side derived from negative qty, backtest direction='short', inverted OCO."""
        mock_config.BACKTEST_START_DATE = datetime(2026, 5, 1)
        mock_config.STOP_LOSS_PCT = 0.05
        mock_config.TAKE_PROFIT_PCT = 0.15

        self.data.get_current_positions_df.return_value = pd.DataFrame({
            'symbol': ['SQQQ'],
            'qty': [-50.0],  # negative = short
            'avg_entry_price': [30.0],
            'current_price': [28.0],
            'market_value': [1400.0],
        })

        # Cloud has a different symbol so SQQQ triggers Alpaca-only path
        short_cloud_df = pd.DataFrame({
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

        def _short_cloud_side_effect(is_open):
            if is_open:
                return short_cloud_df.copy()
            return self._empty_cloud_df()
        self.cloud.get_latest_positions_df.side_effect = _short_cloud_side_effect

        # Mock order history for the short
        entry_submitted = datetime(2026, 5, 20, 9, 30, 0)
        self.data.get_entry_order_for_symbol.return_value = (
            entry_submitted, 30.50
        )

        with patch('optimizer.StrategyOptimizer') as mock_opt_class:
            mock_opt = Mock()
            mock_opt_class.return_value = mock_opt

            backtest_result = BacktestResult(
                symbol='SQQQ',
                rsi_period=10,
                rsi_lower=40,
                rsi_upper=60,
                total_return=0.08,
                buy_and_hold_return=-0.02,
                alpha=0.10,
                num_trades=3,
                win_rate=0.67,
                avg_trade_duration=5,
                max_drawdown=0.01,
                sharpe_ratio=2.0,
                profitable=True,
                current_rsi=65.0,
                composite_score=1.8,
                direction='short',
            )
            mock_opt.optimize_symbol.return_value = backtest_result

            open_positions = self.manager.get_and_reconcile_positions()

        # Only SQQQ should be open; MSFT marked broker_closed
        self.assertEqual(len(open_positions), 1)
        sqqq_pos = [p for p in open_positions if p.symbol == 'SQQQ'][0]
        self.assertEqual(sqqq_pos.symbol, 'SQQQ')
        self.assertEqual(sqqq_pos.side, 'short')
        self.assertEqual(sqqq_pos.quantity, -50.0)

        # entry_date/price from order history (date) and Alpaca (price as source of truth)
        self.assertEqual(sqqq_pos.entry_date, entry_submitted)
        self.assertEqual(sqqq_pos.entry_price, 30.0)

        # RSI params from backtest
        self.assertEqual(sqqq_pos.rsi_period, 10)
        self.assertEqual(sqqq_pos.rsi_lower, 40)
        self.assertEqual(sqqq_pos.rsi_upper, 60)

        # Short OCO: stop_loss ABOVE entry (30.50 * 1.05 = 32.025)
        # take_profit BELOW entry (30.50 * 0.85 = 25.925)
        self.assertAlmostEqual(sqqq_pos.stop_loss_price, 32.025, places=4)
        self.assertAlmostEqual(sqqq_pos.take_profit_price, 25.925, places=4)

        # Backtest called with direction='short'
        mock_opt.optimize_symbol.assert_called_once()
        call_args, call_kwargs = mock_opt.optimize_symbol.call_args
        self.assertEqual(call_args[0], 'SQQQ')
        self.assertEqual(call_kwargs['direction'], 'short')

        # MSFT marked broker_closed
        msft_positions = [
            p for p in self.manager.positions if p.symbol == 'MSFT']
        self.assertEqual(len(msft_positions), 1)
        self.assertTrue(msft_positions[0].closed)
        self.assertEqual(msft_positions[0].exit_reason, 'broker_closed')

    def test_reconcile_cloud_only_short_broker_closed(self):
        """Cloud-only SHORT marked broker_closed with correct realized_return formula."""
        self.data.get_current_positions_df.return_value = self._empty_positions_df()

        # Short position: shares=-20, entry=50, current=48
        short_open_df = pd.DataFrame({
            'symbol': ['XYZ'],
            'shares': [-20.0],
            'entry_price': [50.0],
            'current_price': [48.0],
            'position_value': [960.0],
            'current_rsi': [55.0],
            'entry_date': [datetime(2026, 6, 1)],
            'rsi_period': [14],
            'rsi_lower': [30],
            'rsi_upper': [70],
            'alpha': [0.05],
            'stop_loss_price': [52.0],
            'take_profit_price': [45.0],
            'exit_date': [None],
            'exit_price': [pd.NA],
            'realized_return': [pd.NA],
            'exit_reason': [None],
            'closed': [False],
        })

        def _short_cloud_side_effect(is_open):
            if is_open:
                return short_open_df.copy()
            return self._empty_cloud_df()
        self.cloud.get_latest_positions_df.side_effect = _short_cloud_side_effect

        open_positions = self.manager.get_and_reconcile_positions()

        # No open positions — XYZ is cloud-only, marked broker_closed
        self.assertEqual(len(open_positions), 0)
        closed = [p for p in self.manager.positions if p.closed]
        self.assertEqual(len(closed), 1)
        xyz = closed[0]
        self.assertEqual(xyz.symbol, 'XYZ')
        self.assertEqual(xyz.side, 'short')
        self.assertEqual(xyz.exit_reason, 'broker_closed')

        # OCO fallback: current=48, stop=52, take=45 → |48-52|=4 < |48-45|=3? No, 3 < 4
        # take_profit is closer → exit=45
        self.assertEqual(xyz.exit_price, 45.0)

        # Short realized return: (entry - exit) / entry = (50 - 45) / 50 = 0.10
        self.assertAlmostEqual(xyz.realized_return, 0.10, places=4)

    @patch('positions.globalConfig')
    def test_reconcile_initialized_from_alpaca_enrichment(self, mock_config):
        """When cloud is completely empty, Alpaca positions initialize cloud and run enrichment."""
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

        # Cloud is completely empty — triggers initialized_from_alpaca path
        def _empty_cloud_side_effect(is_open):
            return self._empty_cloud_df()
        self.cloud.get_latest_positions_df.side_effect = _empty_cloud_side_effect

        # Mock order history for enrichment
        entry_submitted = datetime(2026, 5, 25, 9, 30, 0)
        self.data.get_entry_order_for_symbol.return_value = (
            entry_submitted, 150.25
        )

        with patch('optimizer.StrategyOptimizer') as mock_opt_class:
            mock_opt = Mock()
            mock_opt_class.return_value = mock_opt

            enrich_result = BacktestResult(
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
                profitable=True,
                current_rsi=35.0,
                composite_score=1.5,
                direction='long',
            )
            mock_opt.optimize_symbol.return_value = enrich_result

            open_positions = self.manager.get_and_reconcile_positions()

        self.assertEqual(len(open_positions), 1)
        aapl = open_positions[0]
        self.assertEqual(aapl.symbol, 'AAPL')
        self.assertFalse(aapl.closed)

        # entry_date from order history
        self.assertEqual(aapl.entry_date, entry_submitted)
        # entry_price from Alpaca (source of truth)
        self.assertEqual(aapl.entry_price, 150.0)
        # RSI from backtest
        self.assertEqual(aapl.rsi_period, 14)
        self.assertEqual(aapl.rsi_lower, 25)
        self.assertEqual(aapl.rsi_upper, 75)

        # Backtest called with constrained window
        mock_opt.optimize_symbol.assert_called_once()
        call_args, _ = mock_opt.optimize_symbol.call_args
        self.assertEqual(call_args[2], entry_submitted - timedelta(days=1))

    def test_reconcile_cloud_only_missing_shares_column(self):
        """Cloud-only position without 'shares' column still marked broker_closed (no crash)."""
        self.data.get_current_positions_df.return_value = self._empty_positions_df()

        # Position missing the 'shares' column entirely
        no_shares_df = pd.DataFrame({
            'symbol': ['MISSING'],
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

        def _no_shares_side_effect(is_open):
            if is_open:
                return no_shares_df.copy()
            return self._empty_cloud_df()
        self.cloud.get_latest_positions_df.side_effect = _no_shares_side_effect

        open_positions = self.manager.get_and_reconcile_positions()

        # Should not crash; position should be marked broker_closed
        self.assertEqual(len(open_positions), 0)
        closed = [p for p in self.manager.positions if p.closed]
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0].symbol, 'MISSING')
        self.assertEqual(closed[0].exit_reason, 'broker_closed')
        # Without shares, defaults to 0, side defaults to 'long'
        self.assertEqual(closed[0].quantity, 0.0)
        self.assertEqual(closed[0].side, 'long')

    def test_reconcile_stale_cloud_mixed_open_closed(self):
        """Cloud has AAPL+MSFT both open; Alpaca only has AAPL. AAPL stays open, MSFT closed."""
        self.data.get_current_positions_df.return_value = pd.DataFrame({
            'symbol': ['AAPL'],
            'qty': [15.0],
            'avg_entry_price': [155.0],
            'current_price': [160.0],
            'market_value': [2400.0],
        })

        mixed_cloud_df = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT'],
            'shares': [10.0, 5.0],
            'entry_price': [150.0, 300.0],
            'current_price': [155.0, 305.0],
            'position_value': [1550.0, 1525.0],
            'current_rsi': [45.0, 55.0],
            'entry_date': [datetime(2026, 5, 10), datetime(2026, 5, 10)],
            'rsi_period': [14, 14],
            'rsi_lower': [30, 30],
            'rsi_upper': [70, 70],
            'alpha': [0.1, 0.2],
            'stop_loss_price': [142.5, 285.0],
            'take_profit_price': [165.0, 330.0],
            'exit_date': [None, None],
            'exit_price': [pd.NA, pd.NA],
            'realized_return': [pd.NA, pd.NA],
            'exit_reason': [None, None],
            'closed': [False, False],
        })

        def _mixed_cloud_side_effect(is_open):
            if is_open:
                return mixed_cloud_df.copy()
            return self._empty_cloud_df()
        self.cloud.get_latest_positions_df.side_effect = _mixed_cloud_side_effect

        open_positions = self.manager.get_and_reconcile_positions()

        # Only AAPL should be open
        self.assertEqual(len(open_positions), 1)
        aapl = open_positions[0]
        self.assertEqual(aapl.symbol, 'AAPL')
        self.assertFalse(aapl.closed)

        # AAPL updated with live Alpaca values
        self.assertEqual(aapl.quantity, 15.0)
        self.assertEqual(aapl.current_price, 160.0)
        # entry_price overwritten from Alpaca because AAPL was in original cloud
        self.assertEqual(aapl.entry_price, 155.0)

        # MSFT marked broker_closed
        msft_positions = [
            p for p in self.manager.positions if p.symbol == 'MSFT']
        self.assertEqual(len(msft_positions), 1)
        self.assertTrue(msft_positions[0].closed)
        self.assertEqual(msft_positions[0].exit_reason, 'broker_closed')
        # OCO fallback: current=305, stop=285, take=330 → stop closer → exit=285
        self.assertEqual(msft_positions[0].exit_price, 285.0)
        # Long realized return: (285 - 300) / 300 = -0.05
        self.assertAlmostEqual(
            msft_positions[0].realized_return, -0.05, places=4)

    def test_reconcile_zero_quantity_alpaca_position(self):
        """Zero-quantity Alpaca position should not crash but is added as phantom (edge case)."""
        self.data.get_current_positions_df.return_value = pd.DataFrame({
            'symbol': ['PHANTOM'],
            'qty': [0.0],
            'avg_entry_price': [50.0],
            'current_price': [50.0],
            'market_value': [0.0],
        })

        phantom_cloud_df = pd.DataFrame({
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

        def _phantom_cloud_side_effect(is_open):
            if is_open:
                return phantom_cloud_df.copy()
            return self._empty_cloud_df()
        self.cloud.get_latest_positions_df.side_effect = _phantom_cloud_side_effect
        self.data.get_entry_order_for_symbol.return_value = None

        with patch('optimizer.StrategyOptimizer') as mock_opt_class:
            mock_opt = Mock()
            mock_opt_class.return_value = mock_opt

            backtest_result = BacktestResult(
                symbol='PHANTOM',
                rsi_period=14,
                rsi_lower=30,
                rsi_upper=70,
                total_return=0.0,
                buy_and_hold_return=0.0,
                alpha=0.0,
                num_trades=0,
                win_rate=0.0,
                avg_trade_duration=0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                profitable=False,
                current_rsi=50.0,
                composite_score=0.0,
                direction='long',
            )
            mock_opt.optimize_symbol.return_value = backtest_result

            open_positions = self.manager.get_and_reconcile_positions()

        # PHANTOM appears as open with 0 shares (known limitation — does not crash)
        # MSFT is cloud-only → broker_closed
        phantom_positions = [
            p for p in open_positions if p.symbol == 'PHANTOM']
        self.assertEqual(len(phantom_positions), 1)
        self.assertEqual(phantom_positions[0].quantity, 0.0)
        self.assertEqual(phantom_positions[0].side, 'long')

    # ------------------------------------------------------------------
    # Smart cloud-only exit detection tests (Fix 3)
    # ------------------------------------------------------------------

    def test_cloud_only_long_detects_oco_take_profit_from_order_history(self):
        """Cloud-only LONG: filled SELL order found → exit_reason=oco_take_profit."""
        self.data.get_current_positions_df.return_value = self._empty_positions_df()

        cloud_df = pd.DataFrame({
            'symbol': ['RFG'],
            'shares': [15.0],
            'entry_price': [62.30],
            'current_price': [60.79],
            'position_value': [911.85],
            'current_rsi': [55.0],
            'entry_date': [datetime(2026, 7, 9)],
            'rsi_period': [14],
            'rsi_lower': [30],
            'rsi_upper': [70],
            'alpha': [0.05],
            'stop_loss_price': [59.18],
            'take_profit_price': [60.82],
            'exit_date': [None],
            'exit_price': [pd.NA],
            'realized_return': [pd.NA],
            'exit_reason': [None],
            'closed': [False],
        })

        def _cloud_side_effect(is_open):
            return cloud_df.copy() if is_open else self._empty_cloud_df()
        self.cloud.get_latest_positions_df.side_effect = _cloud_side_effect

        # Mock a filled SELL at $60.82 (matching take_profit)
        fill_time = datetime(2026, 7, 14, 17, 37, 11)
        self.data.get_filled_orders_for_symbol.return_value = pd.DataFrame({
            'symbol': ['RFG', 'RFG'],
            'side': ['sell', 'buy'],  # sell = close, buy = entry
            'filled_qty': [15.0, 15.0],
            'filled_avg_price': [60.82, 62.30],
            'submitted_at': [fill_time, datetime(2026, 7, 9, 9, 10, 0)],
            'order_type': ['limit', 'market'],
            'status': ['filled', 'filled'],
        })

        open_positions = self.manager.get_and_reconcile_positions()

        self.assertEqual(len(open_positions), 0)
        closed = [
            p for p in self.manager.positions if p.closed and p.symbol == 'RFG']
        self.assertEqual(len(closed), 1)
        rfg = closed[0]
        self.assertEqual(rfg.exit_reason, 'oco_take_profit')
        self.assertEqual(rfg.exit_price, 60.82)
        # exit_date should come from the fill, not datetime.now()
        self.assertEqual(rfg.exit_date, fill_time)
        # Long: (60.82 - 62.30) / 62.30 = -0.0238
        self.assertAlmostEqual(rfg.realized_return, -0.0238, places=3)

    def test_cloud_only_short_detects_oco_stop_loss_from_order_history(self):
        """Cloud-only SHORT: filled BUY (cover) at stop_loss → exit_reason=oco_stop_loss."""
        self.data.get_current_positions_df.return_value = self._empty_positions_df()

        cloud_df = pd.DataFrame({
            'symbol': ['SHORTY'],
            'shares': [-50.0],
            'entry_price': [45.00],
            'current_price': [48.50],
            'position_value': [-2425.0],
            'current_rsi': [75.0],
            'entry_date': [datetime(2026, 7, 10)],
            'rsi_period': [14],
            'rsi_lower': [30],
            'rsi_upper': [70],
            'alpha': [0.03],
            'stop_loss_price': [47.25],   # 5% above entry
            'take_profit_price': [38.25],  # 15% below entry
            'exit_date': [None],
            'exit_price': [pd.NA],
            'realized_return': [pd.NA],
            'exit_reason': [None],
            'closed': [False],
        })

        def _cloud_side_effect(is_open):
            return cloud_df.copy() if is_open else self._empty_cloud_df()
        self.cloud.get_latest_positions_df.side_effect = _cloud_side_effect

        # Mock a filled BUY (cover) at $47.25 — closer to stop_loss than take_profit
        fill_time = datetime(2026, 7, 14, 18, 15, 0)
        self.data.get_filled_orders_for_symbol.return_value = pd.DataFrame({
            'symbol': ['SHORTY', 'SHORTY'],
            'side': ['buy', 'sell'],  # buy = cover, sell = entry (short)
            'filled_qty': [50.0, 50.0],
            'filled_avg_price': [47.25, 45.00],
            'submitted_at': [fill_time, datetime(2026, 7, 10, 14, 30, 0)],
            'order_type': ['stop_limit', 'market'],
            'status': ['filled', 'filled'],
        })

        open_positions = self.manager.get_and_reconcile_positions()

        self.assertEqual(len(open_positions), 0)
        closed = [
            p for p in self.manager.positions if p.closed and p.symbol == 'SHORTY']
        self.assertEqual(len(closed), 1)
        shorty = closed[0]
        self.assertEqual(shorty.side, 'short')
        self.assertEqual(shorty.exit_reason, 'oco_stop_loss')
        self.assertEqual(shorty.exit_price, 47.25)
        self.assertEqual(shorty.exit_date, fill_time)
        # Short: (45 - 47.25) / 45 = -0.05
        self.assertAlmostEqual(shorty.realized_return, -0.05, places=4)

    def test_cloud_only_failed_to_open_no_entry_fill(self):
        """Cloud-only position with order history but NO entry fill → failed_to_open."""
        self.data.get_current_positions_df.return_value = self._empty_positions_df()

        cloud_df = pd.DataFrame({
            'symbol': ['GHOST'],
            'shares': [10.0],
            'entry_price': [100.0],
            'current_price': [100.0],
            'position_value': [1000.0],
            'current_rsi': [50.0],
            'entry_date': [datetime(2026, 7, 14)],
            'rsi_period': [14],
            'rsi_lower': [30],
            'rsi_upper': [70],
            'alpha': [0.0],
            'stop_loss_price': [95.0],
            'take_profit_price': [110.0],
            'exit_date': [None],
            'exit_price': [pd.NA],
            'realized_return': [pd.NA],
            'exit_reason': [None],
            'closed': [False],
        })

        def _cloud_side_effect(is_open):
            return cloud_df.copy() if is_open else self._empty_cloud_df()
        self.cloud.get_latest_positions_df.side_effect = _cloud_side_effect

        # Order history exists but only has CANCELED orders — no fills
        self.data.get_filled_orders_for_symbol.return_value = pd.DataFrame({
            'symbol': ['GHOST', 'GHOST'],
            'side': ['buy', 'sell'],
            'filled_qty': [0.0, 0.0],
            'filled_avg_price': [0.0, 0.0],
            'submitted_at': [datetime(2026, 7, 14, 10, 0), datetime(2026, 7, 14, 10, 1)],
            'order_type': ['market', 'market'],
            'status': ['canceled', 'canceled'],
        })

        open_positions = self.manager.get_and_reconcile_positions()

        self.assertEqual(len(open_positions), 0)
        closed = [
            p for p in self.manager.positions if p.closed and p.symbol == 'GHOST']
        self.assertEqual(len(closed), 1)
        ghost = closed[0]
        self.assertEqual(ghost.exit_reason, 'failed_to_open')
        self.assertEqual(ghost.exit_price, 0.0)

    def test_cloud_only_no_order_history_falls_back_to_broker_closed(self):
        """Cloud-only with no order history at all → broker_closed (unchanged)."""
        self.data.get_current_positions_df.return_value = self._empty_positions_df()

        cloud_df = pd.DataFrame({
            'symbol': ['MYSTERY'],
            'shares': [25.0],
            'entry_price': [80.0],
            'current_price': [82.0],
            'position_value': [2050.0],
            'current_rsi': [45.0],
            'entry_date': [datetime(2026, 7, 8)],
            'rsi_period': [14],
            'rsi_lower': [30],
            'rsi_upper': [70],
            'alpha': [0.04],
            'stop_loss_price': [np.nan],
            'take_profit_price': [np.nan],
            'exit_date': [None],
            'exit_price': [pd.NA],
            'realized_return': [pd.NA],
            'exit_reason': [None],
            'closed': [False],
        })

        def _cloud_side_effect(is_open):
            return cloud_df.copy() if is_open else self._empty_cloud_df()
        self.cloud.get_latest_positions_df.side_effect = _cloud_side_effect

        # Empty order history
        self.data.get_filled_orders_for_symbol.return_value = pd.DataFrame()

        open_positions = self.manager.get_and_reconcile_positions()

        self.assertEqual(len(open_positions), 0)
        closed = [
            p for p in self.manager.positions if p.closed and p.symbol == 'MYSTERY']
        self.assertEqual(len(closed), 1)
        mystery = closed[0]
        self.assertEqual(mystery.exit_reason, 'broker_closed')
        # Fallback exit price: no OCO targets → uses current_price
        self.assertEqual(mystery.exit_price, 82.0)

    # ------------------------------------------------------------------
    # Reconciliation cache tests (performance refinement)
    # ------------------------------------------------------------------

    def test_reconciliation_cache_returns_cached_result_on_second_call(self):
        """Second reconcile within TTL returns cached result without API calls."""
        self.data.get_current_positions_df.return_value = pd.DataFrame({
            'symbol': ['AAPL'],
            'qty': [10.0],
            'avg_entry_price': [100.0],
            'current_price': [101.0],
            'market_value': [1010.0],
        })

        cloud_df = pd.DataFrame({
            'symbol': ['AAPL'],
            'shares': [10.0],
            'entry_price': [100.0],
            'current_price': [101.0],
            'position_value': [1010.0],
            'current_rsi': [45.0],
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
            'exit_reason': [None],
            'closed': [False],
        })

        def _cloud_side_effect(is_open):
            return cloud_df.copy() if is_open else self._empty_cloud_df()
        self.cloud.get_latest_positions_df.side_effect = _cloud_side_effect

        # First call: should hit Alpaca API + cloud storage
        first = self.manager.get_and_reconcile_positions()
        self.assertEqual(len(first), 1)

        api_calls_before = self.data.get_current_positions_df.call_count

        # Second call within TTL: should return cached result, NO new API calls
        second = self.manager.get_and_reconcile_positions()
        self.assertEqual(len(second), 1)
        self.assertEqual(
            self.data.get_current_positions_df.call_count, api_calls_before,
            "Second reconcile should NOT call Alpaca API (should use cache)"
        )

    def test_open_position_invalidates_reconciliation_cache(self):
        """open_position() clears the cache, forcing next reconcile to call API."""
        self.data.get_current_positions_df.return_value = pd.DataFrame({
            'symbol': ['AAPL'],
            'qty': [10.0],
            'avg_entry_price': [100.0],
            'current_price': [101.0],
            'market_value': [1010.0],
        })

        cloud_df = pd.DataFrame({
            'symbol': ['AAPL'],
            'shares': [10.0],
            'entry_price': [100.0],
            'current_price': [101.0],
            'position_value': [1010.0],
            'current_rsi': [45.0],
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
            'exit_reason': [None],
            'closed': [False],
        })

        def _cloud_side_effect(is_open):
            return cloud_df.copy() if is_open else self._empty_cloud_df()
        self.cloud.get_latest_positions_df.side_effect = _cloud_side_effect

        # Populate the cache
        self.manager.get_and_reconcile_positions()
        api_calls_after_first = self.data.get_current_positions_df.call_count

        # Open a new position — this should invalidate the cache
        new_pos = Position(
            symbol="MSFT", quantity=5.0, entry_price=300.0,
            current_price=305.0, current_rsi=50.0,
            entry_date=datetime.now(), alpha=0.02,
            rsi_period=14, rsi_lower=30, rsi_upper=70,
            closed=False,
        )
        self.manager.open_position(new_pos)

        # Next reconcile should call Alpaca API again (cache invalidated)
        self.manager.get_and_reconcile_positions()
        self.assertGreater(
            self.data.get_current_positions_df.call_count, api_calls_after_first,
            "Reconcile after open_position() should call Alpaca API (cache invalidated)"
        )

    def test_invalidate_reconciliation_cache_clears_state(self):
        """invalidate_reconciliation_cache() forces a fresh reconcile."""
        self.data.get_current_positions_df.return_value = pd.DataFrame({
            'symbol': ['AAPL'],
            'qty': [10.0],
            'avg_entry_price': [100.0],
            'current_price': [101.0],
            'market_value': [1010.0],
        })

        cloud_df = pd.DataFrame({
            'symbol': ['AAPL'],
            'shares': [10.0],
            'entry_price': [100.0],
            'current_price': [101.0],
            'position_value': [1010.0],
            'current_rsi': [45.0],
            'entry_date': [datetime.now()],
            'rsi_period': [14],
            'rsi_lower': [30],
            'rsi_upper': [70],
            'alpha': [0.1],
            'stop_loss_price': [np.nan],
            'take_profit_price': [np.nan],
            'exit_date': [None],
            'exit_price': [pd.NA],
            'realized_return': [pd.NA],
            'exit_reason': [None],
            'closed': [False],
        })

        def _cloud_side_effect(is_open):
            return cloud_df.copy() if is_open else self._empty_cloud_df()
        self.cloud.get_latest_positions_df.side_effect = _cloud_side_effect

        # Populate cache
        self.manager.get_and_reconcile_positions()
        api_calls = self.data.get_current_positions_df.call_count
        self.assertIsNotNone(self.manager._reconciled_at)
        self.assertIsNotNone(self.manager._cached_open_positions)

        # Invalidate
        self.manager.invalidate_reconciliation_cache()
        self.assertIsNone(self.manager._reconciled_at)
        self.assertIsNone(self.manager._cached_open_positions)

        # Next call should hit API
        self.manager.get_and_reconcile_positions()
        self.assertGreater(
            self.data.get_current_positions_df.call_count, api_calls)

    def test_find_fill_by_client_order_id(self):
        """Exact match by client_order_id returns price/qty/filled_at."""
        self.data.get_filled_orders_for_symbol.return_value = pd.DataFrame([
            {"symbol": "AAPL", "side": "sell", "filled_qty": 10.0,
             "filled_avg_price": 155.0, "client_order_id": "AAPL-BUY-1",
             "order_id": "o1", "submitted_at": datetime(2025, 6, 10),
             "filled_at": datetime(2025, 6, 10)},
        ])
        result = self.manager._find_fill_by_client_order_id("AAPL", "AAPL-BUY-1")
        self.assertIsNotNone(result)
        price, qty, _ = result
        self.assertEqual(price, 155.0)
        self.assertEqual(qty, 10.0)

    def test_find_fill_by_client_order_id_fallback_to_order_id(self):
        """Falls back to matching by order_id when client_order_id differs."""
        self.data.get_filled_orders_for_symbol.return_value = pd.DataFrame([
            {"symbol": "AAPL", "side": "sell", "filled_qty": 10.0,
             "filled_avg_price": 155.0, "client_order_id": None,
             "order_id": "o1", "submitted_at": datetime(2025, 6, 10),
             "filled_at": datetime(2025, 6, 10)},
        ])
        result = self.manager._find_fill_by_client_order_id("AAPL", "o1")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], 155.0)

    def test_find_fill_by_client_order_id_no_match(self):
        self.data.get_filled_orders_for_symbol.return_value = pd.DataFrame([
            {"symbol": "AAPL", "side": "sell", "filled_qty": 10.0,
             "filled_avg_price": 155.0, "client_order_id": "OTHER",
             "order_id": "o1", "submitted_at": datetime(2025, 6, 10),
             "filled_at": datetime(2025, 6, 10)},
        ])
        self.assertIsNone(
            self.manager._find_fill_by_client_order_id("AAPL", "NOPE"))

    def test_close_position_prefers_client_order_id_fill(self):
        """close_position uses the exact client_order_id fill, not a heuristic."""
        self.data.get_filled_orders_for_symbol.return_value = pd.DataFrame([
            {"symbol": "AAPL", "side": "sell", "filled_qty": 10.0,
             "filled_avg_price": 155.0, "client_order_id": "AAPL-BUY-1",
             "order_id": "o1", "submitted_at": datetime(2025, 6, 10),
             "filled_at": datetime(2025, 6, 10)},
        ])
        p = Position(
            symbol="AAPL", quantity=10.0, entry_price=150.0, current_price=151.0,
            current_rsi=45.0, entry_date=datetime(2025, 6, 1), alpha=0.05,
            rsi_period=14, rsi_lower=30, rsi_upper=70,
            stop_loss_price=140.0, take_profit_price=160.0,
            client_order_id="AAPL-BUY-1",
        )
        self.manager.positions = [p]
        self.cloud.get_latest_positions_df.return_value = pd.DataFrame()

        self.manager.close_position("AAPL")

        self.assertTrue(p.closed)
        self.assertEqual(p.exit_price, 155.0)


if __name__ == '__main__':
    unittest.main()
