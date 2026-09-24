#!/usr/bin/env python3
"""
Unit tests for the main module.
"""
import os
import sys
import unittest
from unittest.mock import AsyncMock, Mock, patch, MagicMock

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))


class TestMainModule(unittest.IsolatedAsyncioTestCase):
    """Test cases for the main module functions."""

    @patch('utils.setup_logging')
    @patch('utils.TradingCalendar')
    def test_main_execution_outside_trading_hours(self, mock_trading_calendar_class, _mock_setup_logging):
        """Test main execution outside trading hours."""
        mock_trading_calendar = Mock()
        mock_trading_calendar.is_trading_day.return_value = False
        mock_trading_calendar_class.return_value = mock_trading_calendar

        with patch('main.logger'):
            # Import and test TradingAlgorithm
            from main import TradingAlgorithm

            algorithm = TradingAlgorithm()

            # The algorithm should handle non-trading days
            self.assertIsNotNone(algorithm)

    @patch('utils.setup_logging')
    @patch('utils.TradingCalendar')
    def test_main_execution_non_trading_day(self, mock_trading_calendar_class, _mock_setup_logging):
        """Test main execution on non-trading day."""
        mock_trading_calendar = Mock()
        mock_trading_calendar.is_trading_day.return_value = False
        mock_trading_calendar_class.return_value = mock_trading_calendar

        with patch('main.logger'):
            # Import and test TradingAlgorithm
            from main import TradingAlgorithm

            algorithm = TradingAlgorithm()

            # The algorithm should handle non-trading days
            self.assertIsNotNone(algorithm)

    @patch('utils.setup_logging')
    @patch('utils.TradingCalendar')
    async def test_main_execution_during_trading_hours(self, mock_trading_calendar_class, _mock_setup_logging):
        """Test main execution during trading hours."""
        mock_trading_calendar = Mock()
        mock_trading_calendar.is_trading_day.return_value = True
        mock_trading_calendar_class.return_value = mock_trading_calendar

        with patch('optimizer.StrategyOptimizer') as mock_optimizer_class:
            with patch('main.TradingEngine') as mock_trading_engine_class:
                # Mock optimizer
                mock_optimizer = Mock()
                mock_optimizer.optimize_universe.return_value = []
                mock_optimizer.filter_results.return_value = []
                mock_optimizer_class.return_value = mock_optimizer

                # Mock trading engine
                mock_trading_engine = Mock()
                mock_trading_engine.execute_trading_session.return_value = {
                    'status': 'completed'}
                mock_trading_engine_class.return_value = mock_trading_engine

                with patch('main.data_provider') as mock_data_provider:
                    mock_data_provider.get_account_info.return_value = {
                        'equity': 10000, 'cash': 5000}
                    mock_data_provider.get_stock_universe.return_value = Mock(
                        empty=False, tolist=lambda: ['AAPL'])

                    # Import and test TradingAlgorithm
                    from main import TradingAlgorithm

                    algorithm = TradingAlgorithm()
                    result = await algorithm.run_full_cycle(force_backtest=True)

                    # Verify that the algorithm executed
                    self.assertIsNotNone(result)

    @patch('utils.setup_logging')
    def test_main_execution_with_exception(self, _mock_setup_logging):
        """Test main execution with exception handling."""
        with patch('main.TradingEngine', side_effect=Exception("Test error")):
            with patch('main.logger'):
                # Import should handle exceptions gracefully
                from main import TradingAlgorithm

                # Should raise exception during initialization
                with self.assertRaises(Exception):
                    TradingAlgorithm()

    @patch('main.globalConfig')
    @patch('main.StrategyOptimizer')
    @patch('main.storage')
    async def test_run_backtests_function(self, mock_storage, mock_optimizer_class, mock_global_config):
        """Test the backtest functionality in TradingAlgorithm."""
        # Disable walk-forward to take the direct optimizer path
        mock_global_config.WF_ENABLED = False
        # Multi-strategy: single enabled strategy (legacy behavior)
        mock_global_config.STRATEGIES_ENABLED = ['rsi_mean_reversion']
        mock_global_config.STRATEGY_ALLOCATION = {}

        # Mock optimizer
        mock_optimizer = Mock()
        mock_result = Mock()
        mock_result.profitable = True
        mock_optimizer.optimize_universe = AsyncMock(
            return_value=[mock_result])
        mock_optimizer.filter_results.return_value = [mock_result]
        mock_optimizer_class.return_value = mock_optimizer

        # Mock storage upload
        mock_storage.save_backtest_results.return_value = True

        with patch('main.data_provider') as mock_data_provider:
            mock_universe_df = MagicMock()
            mock_universe_df.empty = False
            mock_universe_df.__getitem__ = MagicMock(return_value=MagicMock())
            mock_universe_df['symbol'].tolist.return_value = ['AAPL']
            mock_data_provider.get_stock_universe.return_value = mock_universe_df

            # Import the class
            from main import TradingAlgorithm

            algorithm = TradingAlgorithm()
            results = await algorithm._get_backtest_results(force_backtest=True)

            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].profitable)
            mock_optimizer.optimize_universe.assert_called()

    @patch('main.TradingEngine')
    async def test_execute_trades_function(self, mock_trading_engine_class):
        """Test the trading execution functionality in TradingAlgorithm."""
        # Mock trading engine
        mock_trading_engine = Mock()
        mock_trading_engine.execute_trading_session.return_value = {
            'orders_placed': 1, 'total_value': 1000}
        mock_trading_engine_class.return_value = mock_trading_engine

        with patch('main.data_provider') as mock_data_provider:
            mock_data_provider.get_account_info.return_value = {
                'equity': 10000, 'cash': 5000}

            # Import the class
            from main import TradingAlgorithm

            algorithm = TradingAlgorithm()
            mock_backtest_results = [Mock()]

            # Test the trading session execution
            result = algorithm.trading_engine.execute_trading_session(
                mock_backtest_results)

            self.assertIsNotNone(result)
            mock_trading_engine.execute_trading_session.assert_called_with(
                mock_backtest_results)

    @patch('main.globalConfig')
    def test_config_validation(self, mock_global_config):
        """Test configuration validation."""
        # Test with valid config
        mock_global_config.PAPER_TRADE = True
        mock_global_config.MIN_CASH_PCT = 0.1
        mock_global_config.to_dict.return_value = {'paper_trade': True}

        # Import should work without issues
        try:
            from main import TradingAlgorithm  # pylint: disable=unused-import
            config_valid = True
        except Exception:  # pylint: disable=broad-exception-caught
            config_valid = False

        self.assertTrue(config_valid)


class TestBacktestRetention(unittest.TestCase):
    """Retention pruning runs at the end of every backtest pass."""

    def _algorithm(self):
        with patch('main.StrategyOptimizer'), \
                patch('main.PositionsManager'), \
                patch('main.TradingEngine'), \
                patch('main.WalkForwardValidator'), \
                patch('main.start_health_server'), \
                patch('main.data_provider'):
            from main import TradingAlgorithm
            return TradingAlgorithm()

    def _call(self, algorithm, retention_days, prune_return=0, prune_error=None):
        with patch('main.globalConfig') as cfg, \
                patch('main.storage') as mock_storage:
            cfg.BACKTEST_RESULTS_RETENTION_DAYS = retention_days
            if prune_error is not None:
                mock_storage.prune_backtest_results.side_effect = prune_error
            else:
                mock_storage.prune_backtest_results.return_value = prune_return
            algorithm._prune_old_backtest_results()
        return mock_storage

    def test_prunes_with_configured_window(self):
        algorithm = self._algorithm()
        mock_storage = self._call(algorithm, 30, prune_return=12)
        mock_storage.prune_backtest_results.assert_called_once_with(30)

    def test_disabled_when_zero(self):
        algorithm = self._algorithm()
        mock_storage = self._call(algorithm, 0)
        mock_storage.prune_backtest_results.assert_not_called()

    def test_disabled_when_negative(self):
        algorithm = self._algorithm()
        mock_storage = self._call(algorithm, -7)
        mock_storage.prune_backtest_results.assert_not_called()

    def test_disabled_when_attribute_missing(self):
        """An older config without the setting must not crash the cycle."""
        algorithm = self._algorithm()
        with patch('main.globalConfig') as cfg, \
                patch('main.storage') as mock_storage:
            del cfg.BACKTEST_RESULTS_RETENTION_DAYS
            algorithm._prune_old_backtest_results()
        mock_storage.prune_backtest_results.assert_not_called()

    def test_prune_failure_never_raises(self):
        """Retention is best-effort and must not fail a trading cycle."""
        algorithm = self._algorithm()
        self._call(algorithm, 30, prune_error=RuntimeError("db down"))

    def test_malformed_value_is_rejected_not_crashed(self):
        """A non-numeric setting must warn and skip, never raise."""
        algorithm = self._algorithm()
        for bad in ("thirty", object(), None):
            mock_storage = self._call(algorithm, bad)
            mock_storage.prune_backtest_results.assert_not_called()

    def test_numeric_string_value_is_accepted(self):
        algorithm = self._algorithm()
        mock_storage = self._call(algorithm, "30", prune_return=1)
        mock_storage.prune_backtest_results.assert_called_once_with(30)

    def test_bool_value_is_rejected(self):
        """True is an int in Python but is not a valid day count."""
        algorithm = self._algorithm()
        mock_storage = self._call(algorithm, True)
        mock_storage.prune_backtest_results.assert_not_called()


class TestBacktestRetentionAsync(unittest.IsolatedAsyncioTestCase):
    """The prune hook runs on the backtest path, after the results are saved."""

    def _algorithm(self):
        with patch('main.StrategyOptimizer'), \
                patch('main.PositionsManager'), \
                patch('main.TradingEngine'), \
                patch('main.WalkForwardValidator'), \
                patch('main.start_health_server'), \
                patch('main.data_provider'):
            from main import TradingAlgorithm
            return TradingAlgorithm()

    async def test_prune_runs_after_saving_results(self):
        algorithm = self._algorithm()
        calls = []

        with patch('main.globalConfig') as cfg, \
                patch('main.storage') as mock_storage, \
                patch('main.StrategyOptimizer') as opt_cls, \
                patch('main.data_provider') as dp:
            cfg.WF_ENABLED = False
            cfg.STRATEGIES_ENABLED = ['rsi_mean_reversion']
            cfg.BACKTEST_START_DATE = Mock()
            cfg.BACKTEST_RESULTS_RETENTION_DAYS = 30

            optimizer = Mock()
            optimizer.optimize_universe = AsyncMock(return_value=[])
            optimizer.filter_results.return_value = []
            opt_cls.return_value = optimizer

            universe = MagicMock()
            universe.empty = False
            universe.__getitem__ = MagicMock(return_value=MagicMock())
            universe['symbol'].tolist.return_value = ['AAPL']
            dp.get_stock_universe.return_value = universe

            mock_storage.save_backtest_results.side_effect = (
                lambda *a, **k: calls.append('save') or True)
            mock_storage.prune_backtest_results.side_effect = (
                lambda *a, **k: calls.append('prune') or 0)

            await algorithm._get_backtest_results(force_backtest=True)

        self.assertEqual(calls, ['save', 'prune'])


class TestStrategyPerformanceSnapshot(unittest.TestCase):
    """Tests for the per-strategy end-of-session performance snapshot."""

    def test_build_records_and_save(self):
        from main import TradingAlgorithm

        with patch('main.globalConfig') as cfg:
            cfg.STRATEGIES_ENABLED = ['rsi_mean_reversion',
                                      'leveraged_flow_portfolio']
            cfg.STRATEGY_ALLOCATION = {'rsi_mean_reversion': 0.85,
                                       'leveraged_flow_portfolio': 0.15}

            with patch('main.TradingEngine') as engine_cls:
                engine = Mock()
                engine._strategy_budgets.return_value = {
                    'rsi_mean_reversion': 85000.0,
                    'leveraged_flow_portfolio': 15000.0,
                }
                engine_cls.return_value = engine

                def _pos(symbol, qty, entry, current, strategy, closed=False,
                         realized=None):
                    p = Mock()
                    p.symbol = symbol
                    p.quantity = qty
                    p.entry_price = entry
                    p.current_price = current
                    p.strategy_name = strategy
                    p.closed = closed
                    p.realized_return = realized
                    return p

                pm = Mock()
                pm.positions = [
                    _pos('AAPL', 100, 150.0, 155.0, 'rsi_mean_reversion'),
                    _pos('AMD', 50, 100.0, 95.0, 'rsi_mean_reversion'),
                    _pos('TSL3L', -200, 10.0, 10.5,
                         'leveraged_flow_portfolio'),
                    _pos('OLD', 10, 20.0, 22.0, 'rsi_mean_reversion',
                         closed=True, realized=0.05),
                ]

                algorithm = object.__new__(TradingAlgorithm)
                algorithm.positions_manager = pm
                algorithm.trading_engine = engine

                records = algorithm._build_strategy_performance_records(
                    {'equity': 100000.0})

                by_name = {r['strategy_name']: r for r in records}
                self.assertEqual(set(by_name),
                                 {'rsi_mean_reversion',
                                  'leveraged_flow_portfolio'})

                rsi = by_name['rsi_mean_reversion']
                self.assertEqual(rsi['open_positions'], 2)
                # 100*155 + 50*95
                self.assertAlmostEqual(rsi['open_market_value'], 20250.0)
                # (155-150)*100 + (95-100)*50 = 500 - 250
                self.assertAlmostEqual(rsi['unrealized_pnl'], 250.0)
                # 0.05 * 20 * 10
                self.assertAlmostEqual(rsi['realized_pnl'], 10.0)
                self.assertAlmostEqual(rsi['allocation_weight'], 0.85)

                flow = by_name['leveraged_flow_portfolio']
                self.assertEqual(flow['open_positions'], 1)
                self.assertAlmostEqual(flow['open_market_value'], 2100.0)
                # short: (10.5-10.0) * (-200) = -100
                self.assertAlmostEqual(flow['unrealized_pnl'], -100.0)
                self.assertAlmostEqual(flow['realized_pnl'], 0.0)

                # Persisting uses storage.save_strategy_performance
                with patch('main.storage') as mock_storage:
                    algorithm._save_strategy_performance_snapshot(
                        {'equity': 100000.0})
                    mock_storage.save_strategy_performance.assert_called_once()
                    args = mock_storage.save_strategy_performance.call_args[0]
                    self.assertEqual(args[1], records[0]['snapshot_date'])

    def test_snapshot_failure_is_non_fatal(self):
        from main import TradingAlgorithm
        with patch('main.globalConfig') as cfg:
            cfg.STRATEGIES_ENABLED = ['rsi_mean_reversion']
            cfg.STRATEGY_ALLOCATION = {}
            algorithm = object.__new__(TradingAlgorithm)
            pm = Mock()
            pm.positions = None
            algorithm.positions_manager = pm
            engine = Mock()
            engine._strategy_budgets.return_value = {}
            algorithm.trading_engine = engine
            # Should not raise even with broken storage
            with patch('main.storage') as mock_storage:
                mock_storage.save_strategy_performance.side_effect = \
                    RuntimeError('boom')
                algorithm._save_strategy_performance_snapshot({'equity': 1})


if __name__ == '__main__':
    unittest.main()
