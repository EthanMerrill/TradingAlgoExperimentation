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

        # Mock optimizer
        mock_optimizer = Mock()
        mock_result = Mock()
        mock_result.profitable = True
        mock_optimizer.optimize_universe = AsyncMock(
            return_value=[mock_result])
        mock_optimizer.filter_results.return_value = [mock_result]
        mock_optimizer_class.return_value = mock_optimizer

        # Mock cloud storage upload
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


if __name__ == '__main__':
    unittest.main()
